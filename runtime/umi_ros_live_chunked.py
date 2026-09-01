#!/usr/bin/env python3
"""Direct pi0.5 live control for the XR dual-arm robot.

Output is possible only while the explicit enable file exists. This is the
profile-driven H50/H90 policy client. Each action is decoded against that
request's shared observation anchor and sent directly; no cumulative chaining
is used inside a chunk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from async_chunk_scheduler import (
    AsyncChunkBuffer,
    plan_chunk_prefix,
    wall_time_to_monotonic_ns,
)
from camera_sync import (
    CLOCK_DOMAIN,
    CLOCK_SYNC_SCHEMA,
    DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    OBSERVATION_RETRY_SCHEMA,
    OBSERVATION_TRANSPORT_AGE_POLICY,
    SYNC_SCHEMA,
    translate_sync_to_policy_clock,
    validate_camera_sync_response,
    validate_eef_sample_intervals,
    validate_observation_interval,
)
from camera_transport import WRIST_TRANSPORT_SCHEMA
from inference_stop import ExecutionGate, StopRequested, wait_future_interruptibly
from low_latency_policy_client import (
    ClockSyncUncertain,
    CLOCK_SYNC_MAX_UNCERTAINTY_MS,
    LowLatencyWebsocketClientPolicy,
    TRANSPORT_PROBE_SCHEMA,
    estimate_server_clock_offset,
    warm_transport,
)
from se3_actions import (
    build_state30,
    convert_v2_actions_to_current,
    decode_shared_anchor_actions,
    decoded_targets_to_pose7,
    load_v2_to_current_action_bases,
    matrix_to_quaternion_xyzw,
)
from robot_observation import (
    DEFAULT_CAMERA_SYNC_MAX_SKEW_MS,
    HAND_MAX_DEG,
    ObservationNode,
    wait_snapshot,
    wrist_images_for_transport,
)
from robot_command_interpolation import (
    ChunkLeftZExecutionAnchor,
    interpolate_arm_segment,
    next_dispatch_deadline,
)
from umi_live_contract import (
    GENERIC_CONTRACT_SOURCE,
    GENERIC_PROMPT_SOURCE,
    validate_policy_contract_metadata,
    validated_execution_schedule,
    validated_head_camera_preprocess,
)


DEFAULT_LIVE_CHUNK_SIZE = 40
DEFAULT_INFERENCE_LEAD_STEPS = 5
DEFAULT_CONTROL_RATE_HZ = 10.0
DEFAULT_ARM_COMMAND_RATE_HZ = 50.0
CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M = -0.010


SDK_WORK_MODE_SETTLE_SECONDS = 2.0
END_POSE_MODE_SETTLE_SECONDS = 2.0


def validated_contract_action_shape(
    contract: Mapping[str, Any],
) -> tuple[int, int]:
    """Extract an unambiguous positive action shape from a reviewed contract."""

    values: list[int] = []
    for key in ("action_horizon", "action_dim"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"policy contract {key} must be a positive integer; got {value!r}"
            )
        values.append(value)
    return values[0], values[1]


def validate_policy_action_tensor(
    value: Any,
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Decode actions only when their full shape exactly matches the contract."""

    expected_shape = (action_horizon, action_dim)
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"invalid policy action tensor: expected={expected_shape} decode_error={exc}"
        ) from exc
    if actions.shape != expected_shape or not np.all(np.isfinite(actions)):
        raise RuntimeError(
            "invalid policy action tensor: "
            f"expected={expected_shape} actual={actions.shape} finite="
            f"{bool(np.all(np.isfinite(actions)))}"
        )
    return actions


def has_explicit_execution_schedule(contract: Mapping[str, Any]) -> bool:
    """Return whether robot sampling is explicitly owned by the profile."""

    return any(
        key in contract
        for key in (
            "execution_rate_hz",
            "execution_stride",
            "execution_first_index",
        )
    )


def validate_requested_control_rate(
    control_rate_hz: float,
    contract_execution_rate_hz: float,
    *,
    schedule_explicit: bool,
) -> None:
    """Enforce profile-owned rates while preserving legacy selectable rates."""

    for name, value in (
        ("control rate", control_rate_hz),
        ("contract execution rate", contract_execution_rate_hz),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number: {value!r}")

    if schedule_explicit and abs(control_rate_hz - contract_execution_rate_hz) > 1e-9:
        raise ValueError(
            "explicit policy execution schedule requires --rate to match the "
            f"contract: requested={control_rate_hz:g} "
            f"contract={contract_execution_rate_hz:g}"
        )


def positive_control_rate_arg(raw: str) -> float:
    """Argparse converter accepting every positive finite command rate."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"control rate must be a positive finite number, got {raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"control rate must be a positive finite number, got {raw!r}"
        )
    return value


def xr_wall_time_to_monotonic_ns(observation_wall_time_ns: int) -> int:
    """Map one XR-local wall timestamp onto the XR monotonic clock.

    Camera source timestamps are Unix epoch nanoseconds, while asynchronous
    scheduling must not depend on later NTP wall-clock corrections.  Sampling
    both local clocks together gives the monotonic timestamp used for every
    future action-deadline comparison.
    """

    sampled_wall_ns = time.time_ns()
    sampled_monotonic_ns = time.monotonic_ns()
    try:
        return wall_time_to_monotonic_ns(
            observation_wall_time_ns=observation_wall_time_ns,
            sampled_wall_time_ns=sampled_wall_ns,
            sampled_monotonic_ns=sampled_monotonic_ns,
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid XR observation timestamp: {exc}") from exc


@dataclass(frozen=True)
class PolicyChunk:
    sequence: int
    actions: np.ndarray
    left_pose7: np.ndarray
    right_pose7: np.ndarray
    left_hand_deg: np.ndarray
    right_hand_deg: np.ndarray
    input_state30: np.ndarray
    left_anchor_T: np.ndarray
    right_anchor_T: np.ndarray
    policy_ms: float | None
    server_ms: float | None
    recording: dict[str, Any]
    camera_sync: dict[str, Any]
    execution_indices: tuple[int, ...]
    observation_wall_time_ns: int = 0
    observation_monotonic_ns: int = 0
    policy_rate_hz: float = 10.0
    contract_execution_rate_hz: float = 10.0
    schedule_explicit: bool = False
    robot_actions_current: np.ndarray | None = None
    action_output_basis: dict[str, Any] | None = None


@dataclass(frozen=True)
class HandAck:
    generation: int
    received_monotonic: float
    positions: tuple[int, ...]
    requested_positions: tuple[int, ...] | None
    status: str


@dataclass(frozen=True)
class HandPublishReceipt:
    targets: dict[str, tuple[int, ...]]
    published_monotonic: float
    ack_generation_before: dict[str, int]


def require_success(result: Any, label: str) -> None:
    if result is None or not result.is_success:
        raise RuntimeError(f"{label} failed: {getattr(result, 'error_message', '')}")


def degrees_to_motor_positions(degrees: np.ndarray, side: str) -> tuple[int, ...]:
    values = np.asarray(degrees, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{side} hand target must be finite shape (6,)")
    normalized = values / HAND_MAX_DEG.astype(np.float64) * 1000.0
    # Revo2's wire protocol cannot encode values outside 0..1000. Saturate only
    # this final integer representation; the decoded model target remains raw.
    encoded = np.rint(np.clip(normalized, 0.0, 1000.0)).astype(np.int64)
    return tuple(int(value) for value in encoded)


def latest_control_feedback_with_times(
    node: ObservationNode, max_age: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Read fresh control feedback and its monotonic arrival timestamps."""

    required = ("left_eef", "right_eef", "left_hand", "right_hand")
    now = time.monotonic()
    with node.lock:
        missing = [key for key in required if key not in node.values]
        if missing:
            raise RuntimeError(f"missing live control feedback: {missing}")
        stale = [key for key in required if now - node.values[key][0] > max_age]
        if stale:
            raise RuntimeError(f"stale live control feedback: {stale}")
        values = {key: node.values[key][1].copy() for key in required}
        arrivals = {key: float(node.values[key][0]) for key in required}
    return values, arrivals


def latest_control_feedback(
    node: ObservationNode, max_age: float
) -> dict[str, np.ndarray]:
    """Read fresh arm/hand feedback without coupling control to camera timing."""

    values, _arrivals = latest_control_feedback_with_times(node, max_age)
    return values


def wait_control_feedback_after(
    node: ObservationNode,
    *,
    after_monotonic: float,
    timeout: float,
    max_age: float,
    gate: ExecutionGate,
    stage: str,
) -> dict[str, np.ndarray]:
    """Wait until both EEF streams have produced samples after a boundary."""

    deadline = time.monotonic() + timeout
    while True:
        gate.check(stage)
        feedback, arrivals = latest_control_feedback_with_times(node, max_age)
        if all(
            arrivals[side] > after_monotonic
            for side in ("left_eef", "right_eef")
        ):
            return feedback
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for post-boundary EEF feedback during {stage}"
            )
        gate.wait(0.01, stage)


def wait_future_with_command_refresh(
    future: Future[Any],
    *,
    arm: Any,
    gate: ExecutionGate,
    left_hold_pose7: np.ndarray,
    right_hold_pose7: np.ndarray,
    rpc_timeout: float,
    rate_hz: float = DEFAULT_CONTROL_RATE_HZ,
    stream: Any | None = None,
    stage: str,
) -> Any:
    """Wait for inference while refreshing the last complete robot target."""

    left_hold = np.asarray(left_hold_pose7, dtype=np.float64).copy()
    right_hold = np.asarray(right_hold_pose7, dtype=np.float64).copy()
    period = 1.0 / rate_hz
    next_send = time.monotonic()
    while True:
        gate.check(stage)
        if future.done():
            value = future.result()
            gate.check(stage + " completed")
            return value
        now = time.monotonic()
        if now >= next_send:
            dispatch_started = time.monotonic()
            arm.send(left_hold, right_hold, timeout=rpc_timeout)
            dispatch_ms = (time.monotonic() - dispatch_started) * 1000.0
            if stream is not None:
                log_step(
                    stream,
                    {
                        "wall_time": time.time(),
                        "record_type": "command_refresh",
                        "stage": stage,
                        "left_hold_pose7": left_hold.tolist(),
                        "right_hold_pose7": right_hold.tolist(),
                        "dispatch_ms": dispatch_ms,
                    },
                )
            next_send += period
            if next_send <= time.monotonic():
                print(
                    f"COMMAND_REFRESH_OVERRUN stage={stage} "
                    "action=skip_missed_deadline",
                    flush=True,
                )
                next_send = time.monotonic() + period
            continue
        gate.wait(
            min(0.05, next_send - now),
            f"{stage} inference/hold wait",
        )


def assert_executable_chunk_prefix(
    chunk: PolicyChunk,
    chunk_size: int,
    *,
    action_horizon: int,
    execution_indices: tuple[int, ...],
) -> None:
    """Validate the full executable schedule without altering targets."""

    if action_horizon <= 0 or not execution_indices:
        raise RuntimeError("live execution schedule is empty or invalid")
    if tuple(chunk.execution_indices) != tuple(execution_indices):
        raise RuntimeError(
            "policy chunk execution schedule changed after contract validation"
        )
    if not 1 <= chunk_size <= len(execution_indices):
        raise RuntimeError(
            "live chunk size is outside the executable horizon: "
            f"chunk_size={chunk_size} executable_horizon={len(execution_indices)} "
            f"raw_horizon={action_horizon}"
        )
    for execution_index, raw_index in enumerate(execution_indices):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, (int, np.integer))
            or not 0 <= int(raw_index) < action_horizon
        ):
            raise RuntimeError(
                "execution index is outside the raw horizon: "
                f"execution_index={execution_index} raw_index={raw_index!r} "
                f"raw_horizon={action_horizon}"
            )
    for side, targets in (
        ("left", chunk.left_pose7),
        ("right", chunk.right_pose7),
    ):
        if targets.shape != (action_horizon, 7) or not np.all(np.isfinite(targets)):
            raise RuntimeError(
                f"{side} decoded pose targets must be finite shape "
                f"({action_horizon}, 7); actual={targets.shape}"
            )
    for side, targets in (
        ("left", chunk.left_hand_deg),
        ("right", chunk.right_hand_deg),
    ):
        if targets.shape != (action_horizon, 6) or not np.all(np.isfinite(targets)):
            raise RuntimeError(
                f"{side} decoded hand targets must be finite shape "
                f"({action_horizon}, 6); actual={targets.shape}"
            )
    # Asynchronous replacement may use the policy rows after chunk_size as a
    # bridge while the next request is in flight, so fail fast on the complete
    # executable schedule rather than only the planned prefix.
    for raw_index in execution_indices:
        validated_direct_action_targets(chunk, int(raw_index))


def validated_direct_action_targets(
    chunk: PolicyChunk,
    raw_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one decoded target unchanged after format validation."""

    if (
        isinstance(raw_index, bool)
        or not isinstance(raw_index, (int, np.integer))
        or not 0 <= int(raw_index) < len(chunk.left_pose7)
    ):
        raise RuntimeError(f"invalid direct action raw index: {raw_index!r}")
    index = int(raw_index)
    left = np.asarray(chunk.left_pose7[index], dtype=np.float64)
    right = np.asarray(chunk.right_pose7[index], dtype=np.float64)
    left_hand = np.asarray(chunk.left_hand_deg[index], dtype=np.float64)
    right_hand = np.asarray(chunk.right_hand_deg[index], dtype=np.float64)
    for side, pose in (("left", left), ("right", right)):
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"{side} direct arm target must be finite shape (7,)")
        quaternion_norm = float(np.linalg.norm(pose[3:]))
        if abs(quaternion_norm - 1.0) > 1e-5:
            raise RuntimeError(
                f"{side} direct arm quaternion is not normalized: {quaternion_norm}"
            )
    for side, target in (("left", left_hand), ("right", right_hand)):
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise RuntimeError(f"{side} direct hand target must be finite shape (6,)")
        degrees_to_motor_positions(target, side)
    return left.copy(), right.copy(), left_hand.copy(), right_hand.copy()


def latest_hand_feedback(
    node: ObservationNode, max_age: float
) -> dict[str, np.ndarray]:
    """Read fresh hand feedback without depending on unrelated arm topics."""

    required = ("left_hand", "right_hand")
    now = time.monotonic()
    with node.lock:
        missing = [key for key in required if key not in node.values]
        if missing:
            raise RuntimeError(f"missing live hand feedback: {missing}")
        stale = [key for key in required if now - node.values[key][0] > max_age]
        if stale:
            raise RuntimeError(f"stale live hand feedback: {stale}")
        return {key: node.values[key][1].copy() for key in required}


class InferenceWorker:
    """Own one WebSocket policy client and use it from one worker thread."""

    def __init__(
        self,
        node: ObservationNode,
        *,
        host: str,
        port: int,
        timeout: float,
        max_age: float,
        control_rate: float,
        chunk_size: int,
        client_run_id: str,
        camera_sync_max_skew_ms: float,
        gate: ExecutionGate,
        action_basis_config: Path | None = None,
    ) -> None:
        self.node = node
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_age = max_age
        self.control_rate = float(control_rate)
        self.chunk_size = int(chunk_size)
        self.policy_rate: float | None = None
        self.expected_action_horizon: int | None = None
        self.expected_action_dim: int | None = None
        self.expected_execution_schedule: dict[str, Any] | None = None
        self.execution_indices: tuple[int, ...] = ()
        self.schedule_explicit: bool | None = None
        self.client_run_id = client_run_id
        self.camera_sync_max_skew_ms = float(camera_sync_max_skew_ms)
        self.action_output_basis_mode: str | None = None
        self.v2_to_current_action_bases: dict[str, np.ndarray] | None = None
        self.action_basis_config = action_basis_config
        self.action_basis_config_sha256: str | None = None
        self.gate = gate
        self.policy: Any | None = None
        self._policy_lock = threading.Lock()
        self.sequence = 0
        self.prompt: str | None = None

    def abort_io(self) -> None:
        """Unblock a synchronous WebSocket recv from the stop path."""

        with self._policy_lock:
            policy = self.policy
        connection = None if policy is None else getattr(policy, "_ws", None)
        close_socket = None if connection is None else getattr(connection, "close_socket", None)
        if callable(close_socket):
            try:
                close_socket()
            except Exception:
                pass

    def _estimate_clock_until_precise(
        self,
        policy: Any,
        request_role: str,
    ) -> dict[str, Any]:
        """Wait for a usable clock proof without turning Wi-Fi jitter fatal."""

        retry = 0
        while True:
            self.gate.check(f"{request_role} clock synchronization")
            try:
                return estimate_server_clock_offset(policy)
            except ClockSyncUncertain as exc:
                retry += 1
                print(
                    "CLOCK_SYNC_RETRY "
                    f"attempt={retry} "
                    f"uncertainty_ms={exc.uncertainty_ms:.3f} "
                    f"limit_ms={exc.max_uncertainty_ms:.3f} "
                    f"best_rtt_ms={exc.best_rtt_ms:.3f} "
                    "action=wait_for_precise_sample",
                    flush=True,
                )
                self.gate.wait(
                    0.05,
                    f"{request_role} clock synchronization retry",
                )

    def _request_once(
        self,
        policy: Any,
        request_role: str,
    ) -> dict[str, Any]:
        """Capture one new two-frame observation and send exactly one request."""

        # Startup or a prior retry may wait for ROS/camera histories. Discard a
        # readiness snapshot before measuring the cross-host clock; neither
        # model state sample is retained until all probes have completed.
        readiness = wait_snapshot(
            self.node,
            self.timeout,
            self.max_age,
            stop_check=self.gate.check,
        )
        clock_sync = self._estimate_clock_until_precise(policy, request_role)
        self.gate.check(f"{request_role} clock synchronization complete")
        print(
            "CLOCK_SYNC "
            f"server_minus_xr_ms="
            f"{clock_sync['server_minus_client_wall_time_ns'] / 1e6:+.3f} "
            f"uncertainty_ms={clock_sync['uncertainty_ns'] / 1e6:.3f} "
            f"best_rtt_ms={clock_sync['best_round_trip_ns'] / 1e6:.3f} "
            f"samples={clock_sync['sample_count']}",
            flush=True,
        )

        self.sequence += 1
        request_index = self.sequence
        first = wait_snapshot(
            self.node,
            self.timeout,
            self.max_age,
            after_sync=readiness["_sync"],
            stop_check=self.gate.check,
        )
        if self.policy_rate is None:
            raise RuntimeError("policy rate was not selected from server metadata")
        expected_interval_ms = 1000.0 / self.policy_rate
        interval_tolerance_ms = expected_interval_ms * 0.45
        self.gate.wait(
            1.0 / self.policy_rate,
            f"{request_role} {self.policy_rate:g}Hz state interval",
        )
        second = wait_snapshot(
            self.node,
            min(2.0, self.timeout),
            self.max_age,
            after_sync=first["_sync"],
            stop_check=self.gate.check,
        )
        observation_wall_time_ns = int(
            second["_sync"]["observation_wall_time_ns"]
        )
        observation_monotonic_ns = xr_wall_time_to_monotonic_ns(
            observation_wall_time_ns
        )
        state_interval_ms = validate_observation_interval(
            first["_sync"],
            second["_sync"],
            expected_ms=expected_interval_ms,
            tolerance_ms=interval_tolerance_ms,
        )
        eef_intervals_ms = validate_eef_sample_intervals(
            first["_state_timing"],
            second["_state_timing"],
            expected_ms=expected_interval_ms,
            tolerance_ms=interval_tolerance_ms,
        )
        second["_sync"]["state_first_observation_wall_time_ns"] = int(
            first["_sync"]["observation_wall_time_ns"]
        )
        second["_sync"]["state_observation_interval_ms"] = state_interval_ms
        second["_sync"]["state_samples"] = {
            "first": first["_state_timing"],
            "second": second["_state_timing"],
            "eef_intervals_ms": eef_intervals_ms,
        }
        state = build_state30(
            first["left_eef"],
            second["left_eef"],
            first["right_eef"],
            second["right_eef"],
            second["left_hand"],
            second["right_hand"],
        )
        self.gate.check(f"{request_role} websocket send")
        policy_sync = translate_sync_to_policy_clock(
            {
                **second["_sync"],
                "client_send_wall_time_ns": time.time_ns(),
            },
            clock_sync,
        )
        if self.prompt is None:
            raise RuntimeError("policy task prompt was not selected from metadata")
        if self.action_output_basis_mode is None:
            raise RuntimeError("policy action basis was not selected from metadata")
        response = policy.infer(
            {
                "images": wrist_images_for_transport(second),
                "state": state,
                "prompt": self.prompt,
                "_recording": {
                    "client_run_id": self.client_run_id,
                    "client_request_index": request_index,
                    "client_request_role": request_role,
                    "action_output_basis": self.action_output_basis_mode,
                    "action_basis_config_sha256": (
                        self.action_basis_config_sha256
                    ),
                },
                "_sync": policy_sync,
            }
        )
        self.gate.check(f"{request_role} websocket response")
        if not isinstance(response, dict):
            raise RuntimeError(
                f"policy response must be a mapping, got {type(response).__name__}"
            )
        return {
            "response": response,
            "state": state,
            "second": second,
            "policy_sync": policy_sync,
            "clock_sync": clock_sync,
            "request_index": request_index,
            "observation_wall_time_ns": observation_wall_time_ns,
            "observation_monotonic_ns": observation_monotonic_ns,
        }

    def _request_until_aligned(
        self,
        policy: Any,
        request_role: str,
    ) -> dict[str, Any]:
        """Discard transient camera failures and recapture until alignment holds."""

        retry_count = 0
        retryable_reasons = {
            "clock_sync_uncertain",
            "e6_unavailable",
            "e6_stream_stale",
            "three_camera_not_aligned",
        }
        while True:
            attempt = self._request_once(policy, request_role)
            response = attempt["response"]
            retry = response.get("observation_retry")
            if retry is None:
                return attempt
            allowed_retry_response_keys = {"observation_retry", "server_timing"}
            if (
                not set(response).issubset(allowed_retry_response_keys)
                or not isinstance(retry, dict)
                or (
                    "server_timing" in response
                    and not isinstance(response["server_timing"], dict)
                )
            ):
                raise RuntimeError(f"malformed observation retry response: {response}")
            reason = retry.get("reason")
            if (
                retry.get("schema") != OBSERVATION_RETRY_SCHEMA
                or reason not in retryable_reasons
            ):
                raise RuntimeError(f"untrusted observation retry response: {retry}")
            retry_count += 1
            print(
                f"OBSERVATION_RETRY attempt={retry_count} reason={reason} "
                f"message={retry.get('message')} "
                f"transport_age_ms={retry.get('observation_transport_age_ms')} "
                "action=discard_and_recapture",
                flush=True,
            )
            # Never reuse the rejected wrist/state sample. The next iteration
            # remeasures the clock and captures a completely new pair. This
            # wait is deliberately short and checks Ctrl+C every time.
            self.gate.wait(0.01, f"{request_role} observation retry")

    def __call__(self, request_role: str = "inference") -> PolicyChunk:
        self.gate.check(f"{request_role} start")
        with self._policy_lock:
            policy = self.policy
        if policy is None:
            policy = LowLatencyWebsocketClientPolicy(
                host=self.host, port=self.port
            )
            with self._policy_lock:
                self.policy = policy
            self.gate.check(f"{request_role} policy connection")
            metadata = policy.get_server_metadata()
            if metadata.get("e6_eye") != "right":
                raise RuntimeError(f"policy server E6 eye is not right: {metadata}")
            try:
                contract = validate_policy_contract_metadata(metadata)
            except ValueError as exc:
                raise RuntimeError(
                    f"policy deployment contract rejected: {exc}"
                ) from exc
            contract_action_shape = validated_contract_action_shape(contract)
            contract_schedule = validated_execution_schedule(contract)
            schedule_explicit = has_explicit_execution_schedule(contract)
            try:
                validate_requested_control_rate(
                    self.control_rate,
                    float(contract_schedule["execution_rate_hz"]),
                    schedule_explicit=schedule_explicit,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"policy execution rate rejected: {exc}"
                ) from exc
            if self.chunk_size > int(contract_schedule["executable_horizon"]):
                raise RuntimeError(
                    "live chunk size exceeds the executable policy horizon: "
                    f"chunk_size={self.chunk_size} executable_horizon="
                    f"{contract_schedule['executable_horizon']} raw_horizon="
                    f"{contract_action_shape[0]}"
                )
            self.expected_action_horizon, self.expected_action_dim = (
                contract_action_shape
            )
            self.expected_execution_schedule = dict(contract_schedule)
            self.execution_indices = tuple(contract_schedule["indices"])
            self.policy_rate = float(contract_schedule["policy_rate_hz"])
            self.schedule_explicit = schedule_explicit
            head_preprocess = validated_head_camera_preprocess(contract)
            expected_eef_basis = str(contract["eef_basis_mode"])
            expected_action_basis = str(contract["action_output_basis"])
            if expected_eef_basis != "identity":
                raise RuntimeError(
                    "this UMI live client only accepts reviewed identity EEF "
                    f"profiles, got {expected_eef_basis!r}"
                )
            if expected_action_basis == "v2_to_current":
                if self.v2_to_current_action_bases is None:
                    if self.action_basis_config is None:
                        raise RuntimeError(
                            "checkpoint profile requires a local v2_to_current "
                            "basis config path"
                        )
                    try:
                        self.v2_to_current_action_bases = (
                            load_v2_to_current_action_bases(
                                self.action_basis_config
                            )
                        )
                        self.action_basis_config_sha256 = hashlib.sha256(
                            self.action_basis_config.read_bytes()
                        ).hexdigest()
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            "checkpoint profile requires a valid local "
                            f"v2_to_current basis config: {exc}"
                        ) from exc
                if self.v2_to_current_action_bases is None:
                    raise RuntimeError(
                        "checkpoint profile requires the local v2_to_current bases"
                    )
                expected_sha = contract.get("action_basis_config_sha256")
                if (
                    not isinstance(expected_sha, str)
                    or self.action_basis_config_sha256 != expected_sha
                ):
                    raise RuntimeError(
                        "local v2_to_current mapping hash does not match the "
                        f"reviewed profile: expected={expected_sha!r} "
                        f"actual={self.action_basis_config_sha256!r}"
                    )
            elif expected_action_basis != "identity":
                raise RuntimeError(
                    f"unsupported reviewed action basis: {expected_action_basis!r}"
                )
            self.action_output_basis_mode = expected_action_basis
            self.prompt = str(contract["prompt"])
            expected_prompt_source = (
                GENERIC_PROMPT_SOURCE
                if contract.get("contract_source") == GENERIC_CONTRACT_SOURCE
                else "checkpoint_profile"
            )
            if metadata.get("prompt_source") != expected_prompt_source:
                raise RuntimeError(
                    "policy server does not enforce the deployment task prompt"
                )
            print(
                "POLICY_CONTRACT_READY "
                f"task={contract['task_id']} profile={contract['profile_id']} "
                f"id={contract['id']} checkpoint={metadata.get('checkpoint')} "
                f"asset_id={metadata.get('asset_id')} "
                f"action_shape={contract_action_shape} "
                f"policy_rate_hz={contract_schedule['policy_rate_hz']:g} "
                f"contract_execution_rate_hz="
                f"{contract_schedule['execution_rate_hz']:g} "
                f"execution_stride={contract_schedule['stride']} "
                f"execution_first_index={contract_schedule['first_index']} "
                f"executable_horizon="
                f"{contract_schedule['executable_horizon']} "
                f"schedule_explicit={schedule_explicit} "
                f"cam_high_mode={metadata['cam_high_mode_effective']} "
                f"cam_high_preprocess={head_preprocess['mode']} "
                f"action_mapping={expected_action_basis}+"
                f"{contract['eef_anchor']}+{contract['se3_composition']} "
                f"prompt={self.prompt!r}",
                flush=True,
            )
            if metadata.get("e6_frame_live") is not True:
                raise RuntimeError(f"execution requires a live E6 frame: {metadata}")
            if metadata.get("live_actuation_allowed") is not True:
                raise RuntimeError(f"policy server forbids live actuation: {metadata}")
            if metadata.get("recording_enabled") is not True:
                raise RuntimeError(
                    "execution requires the policy I/O recorder, but the server "
                    f"does not report recording_enabled=true: {metadata}"
                )
            if metadata.get("camera_sync_required") is not True:
                raise RuntimeError(
                    f"policy server does not enforce camera synchronization: {metadata}"
                )
            if metadata.get("camera_sync_schema") != SYNC_SCHEMA:
                raise RuntimeError(
                    "policy camera synchronization protocol mismatch: "
                    f"expected={SYNC_SCHEMA!r} actual={metadata.get('camera_sync_schema')!r}"
                )
            if metadata.get("camera_sync_clock_domain") != CLOCK_DOMAIN:
                raise RuntimeError(
                    "policy camera clock domain mismatch: "
                    f"expected={CLOCK_DOMAIN!r} "
                    f"actual={metadata.get('camera_sync_clock_domain')!r}"
                )
            if metadata.get("camera_clock_sync_required") is not True:
                raise RuntimeError("policy server does not require XR/5090 clock correction")
            if metadata.get("camera_clock_sync_schema") != CLOCK_SYNC_SCHEMA:
                raise RuntimeError(
                    "policy camera clock protocol mismatch: "
                    f"expected={CLOCK_SYNC_SCHEMA!r} "
                    f"actual={metadata.get('camera_clock_sync_schema')!r}"
                )
            if (
                metadata.get("camera_clock_sync_age_policy")
                != OBSERVATION_TRANSPORT_AGE_POLICY
            ):
                raise RuntimeError(
                    "policy clock-proof age behavior mismatch: "
                    f"{metadata.get('camera_clock_sync_age_policy')!r}"
                )
            server_clock_uncertainty_limit = float(
                metadata.get(
                    "camera_clock_sync_max_uncertainty_ms", float("inf")
                )
            )
            if (
                not np.isfinite(server_clock_uncertainty_limit)
                or server_clock_uncertainty_limit
                > CLOCK_SYNC_MAX_UNCERTAINTY_MS + 1e-9
            ):
                raise RuntimeError(
                    "policy clock uncertainty limit is missing or weaker than XR: "
                    f"server={server_clock_uncertainty_limit}ms "
                    f"xr={CLOCK_SYNC_MAX_UNCERTAINTY_MS}ms"
                )
            if metadata.get("wrist_image_transport_schema") != WRIST_TRANSPORT_SCHEMA:
                raise RuntimeError(
                    "policy wrist transport protocol mismatch: "
                    f"expected={WRIST_TRANSPORT_SCHEMA!r} "
                    f"actual={metadata.get('wrist_image_transport_schema')!r}"
                )
            if metadata.get("transport_probe_schema") != TRANSPORT_PROBE_SCHEMA:
                raise RuntimeError(
                    "policy transport probe protocol mismatch: "
                    f"expected={TRANSPORT_PROBE_SCHEMA!r} "
                    f"actual={metadata.get('transport_probe_schema')!r}"
                )
            server_limit = float(metadata.get("camera_sync_max_skew_ms", float("nan")))
            if (
                not np.isfinite(server_limit)
                or server_limit <= 0.0
                or server_limit > self.camera_sync_max_skew_ms + 1e-9
            ):
                raise RuntimeError(
                    "policy camera sync limit is weaker than the XR limit: "
                    f"server={server_limit}ms xr={self.camera_sync_max_skew_ms}ms"
                )
            server_state_limit = float(
                metadata.get("state_sync_max_skew_ms", float("nan"))
            )
            if (
                not np.isfinite(server_state_limit)
                or server_state_limit <= 0.0
                or server_state_limit > DEFAULT_STATE_SYNC_MAX_SKEW_MS + 1e-9
            ):
                raise RuntimeError(
                    "policy state sync limit is weaker than the XR limit: "
                    f"server={server_state_limit}ms "
                    f"xr={DEFAULT_STATE_SYNC_MAX_SKEW_MS}ms"
                )
            if (
                metadata.get("camera_sync_observation_transport_age_policy")
                != OBSERVATION_TRANSPORT_AGE_POLICY
            ):
                raise RuntimeError(
                    "policy observation transport-age behavior mismatch: "
                    f"expected={OBSERVATION_TRANSPORT_AGE_POLICY!r} "
                    f"actual={metadata.get('camera_sync_observation_transport_age_policy')!r}"
                )
            if metadata.get("observation_retry_schema") != OBSERVATION_RETRY_SCHEMA:
                raise RuntimeError(
                    "policy observation-retry protocol mismatch: "
                    f"expected={OBSERVATION_RETRY_SCHEMA!r} "
                    f"actual={metadata.get('observation_retry_schema')!r}"
                )
            for probe in warm_transport(policy):
                print(
                    "TRANSPORT_WARMUP "
                    f"probe={probe['sequence']} "
                    f"bytes={probe['payload_bytes']} "
                    f"rtt_ms={probe['round_trip_ms']:.1f}",
                    flush=True,
                )

        if self.action_output_basis_mode not in {"identity", "v2_to_current"}:
            raise RuntimeError("reviewed policy action basis was not selected")
        attempt = self._request_until_aligned(policy, request_role)
        response = attempt["response"]
        state = attempt["state"]
        second = attempt["second"]
        policy_sync = attempt["policy_sync"]
        clock_sync = attempt["clock_sync"]
        request_index = attempt["request_index"]
        observation_wall_time_ns = attempt["observation_wall_time_ns"]
        observation_monotonic_ns = attempt["observation_monotonic_ns"]
        if (
            self.expected_action_horizon is None
            or self.expected_action_dim is None
            or self.expected_execution_schedule is None
            or self.policy_rate is None
            or self.schedule_explicit is None
        ):
            raise RuntimeError("policy contract was not selected before inference")
        if "actions" not in response:
            raise RuntimeError("policy response has no actions tensor")
        actions = validate_policy_action_tensor(
            response["actions"],
            action_horizon=self.expected_action_horizon,
            action_dim=self.expected_action_dim,
        )
        clock_uncertainty_ms = clock_sync["uncertainty_ns"] / 1e6
        camera_sync = validate_camera_sync_response(
            response.get("camera_sync"),
            expected_observation_wall_time_ns=int(
                policy_sync["observation_wall_time_ns"]
            ),
            clock_uncertainty_ms=clock_uncertainty_ms,
            max_camera_skew_ms=self.camera_sync_max_skew_ms,
            expected_observation_interval_ms=1000.0 / self.policy_rate,
            observation_interval_tolerance_ms=(1000.0 / self.policy_rate) * 0.45,
        )
        robot_actions = actions
        action_output_basis: dict[str, Any] = {
            "mode": "identity",
            "model_output_basis": "current",
            "robot_execution_basis": "current",
            "applied_once": False,
        }
        if self.action_output_basis_mode == "v2_to_current":
            if self.v2_to_current_action_bases is None:
                raise RuntimeError("v2_to_current action bases disappeared")
            robot_actions = convert_v2_actions_to_current(
                actions, self.v2_to_current_action_bases
            )
            action_output_basis = {
                "mode": "v2_to_current",
                "model_output_basis": "v2",
                "robot_execution_basis": "current",
                "translation": "p_current=A.T@p_v2",
                "rotation": "dR_current=A.T@dR_v2@A",
                "hands": "unchanged",
                "applied_once": True,
                "config": (
                    None
                    if self.action_basis_config is None
                    else str(self.action_basis_config)
                ),
                "config_sha256": self.action_basis_config_sha256,
                "R_v2_from_current": {
                    side: self.v2_to_current_action_bases[side].tolist()
                    for side in ("left", "right")
                },
            }
        # The reviewed profiles all use current-EEF body deltas with one shared
        # observation anchor: target[k] = T_observation @ Delta[k].
        decoded = decode_shared_anchor_actions(
            robot_actions,
            second["left_eef"],
            second["right_eef"],
            composition="right",
        )
        targets = decoded_targets_to_pose7(decoded)
        recording = response.get("recording")
        if not isinstance(recording, dict) or recording.get("status") != "success":
            raise RuntimeError(f"policy response has no committed recording: {recording}")
        if (
            recording.get("client_run_id") != self.client_run_id
            or recording.get("client_request_index") != request_index
        ):
            raise RuntimeError(
                "policy recording correlation mismatch: "
                f"expected=({self.client_run_id!r},{request_index}) actual={recording}"
            )
        return PolicyChunk(
            sequence=request_index,
            actions=actions,
            left_pose7=targets["left_target_pose7"],
            right_pose7=targets["right_target_pose7"],
            left_hand_deg=targets["left_hand_deg"],
            right_hand_deg=targets["right_hand_deg"],
            input_state30=state,
            left_anchor_T=second["left_eef"].copy(),
            right_anchor_T=second["right_eef"].copy(),
            policy_ms=response.get("policy_timing", {}).get("infer_ms"),
            server_ms=response.get("server_timing", {}).get("infer_ms"),
            recording=dict(recording),
            camera_sync=dict(camera_sync),
            execution_indices=self.execution_indices,
            observation_wall_time_ns=int(observation_wall_time_ns),
            observation_monotonic_ns=int(observation_monotonic_ns),
            policy_rate_hz=self.policy_rate,
            contract_execution_rate_hz=float(
                self.expected_execution_schedule["execution_rate_hz"]
            ),
            schedule_explicit=self.schedule_explicit,
            robot_actions_current=np.asarray(robot_actions, dtype=np.float32),
            action_output_basis=action_output_basis,
        )


class HandCommander:
    def __init__(
        self, node: ObservationNode, *, gate: ExecutionGate | None = None
    ) -> None:
        self._gate = gate
        self._check("Revo2 commander construction")
        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publishers = {
            side: node.create_publisher(
                Float64MultiArray, f"/revo2/absolute/{side}", command_qos
            )
            for side in ("left", "right")
        }
        self._condition = threading.Condition()
        self._ack_generation = {side: 0 for side in ("left", "right")}
        self._acks: dict[str, HandAck] = {}
        self._published_any_target = False
        self._subscriptions = [
            node.create_subscription(
                String,
                f"/revo2/absolute_ack/{side}",
                lambda message, side=side: self._ack(side, message.data),
                command_qos,
            )
            for side in ("left", "right")
        ]

    def _check(self, stage: str) -> None:
        if self._gate is not None:
            self._gate.check(stage)

    def _ack(self, side: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
            if payload.get("schema") != "revo2_absolute_joint_command_v1":
                raise ValueError("bad schema")
            if payload.get("side") != side:
                raise ValueError("bad side")
            # The Revo2 bridge reports ``suppressed`` when the newly requested
            # motor vector differs by less than its one-count transport
            # threshold.  This is a healthy ACK: hardware keeps the last
            # applied target, and the command link is still alive.
            if payload.get("status") not in {"submitted", "unchanged", "suppressed"}:
                raise ValueError(f"bad status {payload.get('status')!r}")
            positions = tuple(int(value) for value in payload["positions"])
            if len(positions) != 6:
                raise ValueError("bad positions")
            requested_raw = payload.get("requested_positions")
            requested_positions = (
                None
                if requested_raw is None
                else tuple(int(value) for value in requested_raw)
            )
            if requested_positions is not None and len(requested_positions) != 6:
                raise ValueError("bad requested_positions")
            with self._condition:
                self._ack_generation[side] += 1
                self._acks[side] = HandAck(
                    generation=self._ack_generation[side],
                    received_monotonic=time.monotonic(),
                    positions=positions,
                    requested_positions=requested_positions,
                    status=payload["status"],
                )
                self._condition.notify_all()
        except Exception as exc:
            raise RuntimeError(f"invalid Revo2 ACK for {side}: {exc}") from exc

    def wait_for_bridge(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check("Revo2 bridge discovery")
            command_ready = all(
                publisher.get_subscription_count() > 0
                for publisher in self.publishers.values()
            )
            ack_ready = all(
                subscription.get_publisher_count() > 0
                for subscription in self._subscriptions
            )
            if command_ready and ack_ready:
                return
            time.sleep(0.05)
        raise RuntimeError("Revo2 absolute command/ACK bridge is not bidirectionally ready")

    def assert_ack_fresh(self, max_age: float) -> None:
        self._check("Revo2 ACK freshness check")
        with self._condition:
            acks = dict(self._acks)
        now = time.monotonic()
        for side in ("left", "right"):
            if (
                side not in acks
                or now - acks[side].received_monotonic > max_age
            ):
                raise RuntimeError(f"lost fresh Revo2 ACK for {side}")

    @staticmethod
    def _ack_matches(
        ack: HandAck, expected: tuple[int, ...], receipt: HandPublishReceipt, side: str
    ) -> bool:
        if ack.generation <= receipt.ack_generation_before[side]:
            return False
        if ack.received_monotonic < receipt.published_monotonic:
            return False
        return ack.positions == expected or ack.requested_positions == expected

    def wait_for_receipt(
        self,
        receipt: HandPublishReceipt,
        timeout: float,
        *,
        enforce_gate: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if enforce_gate:
                    self._check("Revo2 command ACK wait")
                missing = [
                    side
                    for side in ("left", "right")
                    if side not in self._acks
                    or not self._ack_matches(
                        self._acks[side], receipt.targets[side], receipt, side
                    )
                ]
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "timed out waiting for matching Revo2 ACK: "
                        + ",".join(missing)
                    )
                self._condition.wait(timeout=min(0.05, remaining))

    def send(self, left_deg: np.ndarray, right_deg: np.ndarray) -> HandPublishReceipt:
        self._check("Revo2 command encoding")
        targets = {
            "left": degrees_to_motor_positions(left_deg, "left"),
            "right": degrees_to_motor_positions(right_deg, "right"),
        }
        with self._condition:
            generation_before = dict(self._ack_generation)
        published_monotonic = time.monotonic()
        for side, values in targets.items():
            self._check(f"{side} Revo2 publish")
            message = Float64MultiArray()
            message.data = [float(value) for value in values]
            self.publishers[side].publish(message)
            self._published_any_target = True
        return HandPublishReceipt(
            targets=targets,
            published_monotonic=published_monotonic,
            ack_generation_before=generation_before,
        )

    @property
    def has_published_target(self) -> bool:
        return self._published_any_target

    def hold_feedback(
        self, left_deg: np.ndarray, right_deg: np.ndarray, *, timeout: float
    ) -> None:
        """Best-effort no-motion hand target after the execution gate closes."""

        targets = {
            "left": degrees_to_motor_positions(left_deg, "left"),
            "right": degrees_to_motor_positions(right_deg, "right"),
        }
        with self._condition:
            generation_before = dict(self._ack_generation)
        published_monotonic = time.monotonic()
        for side, values in targets.items():
            if self._gate is not None and self._gate.force_emergency:
                raise RuntimeError("second stop signal interrupted Revo2 soft hold")
            message = Float64MultiArray()
            message.data = [float(value) for value in values]
            self.publishers[side].publish(message)
        receipt = HandPublishReceipt(
            targets=targets,
            published_monotonic=published_monotonic,
            ack_generation_before=generation_before,
        )
        self.wait_for_receipt(receipt, timeout=timeout, enforce_gate=False)


def zero_hands_direct_before_live(
    hand: HandCommander,
    node: ObservationNode,
    gate: ExecutionGate,
    *,
    max_age: float,
    timeout: float,
    tolerance_deg: float = 1.0,
) -> dict[str, np.ndarray]:
    """Send one all-zero hand target, without rate limiting, and prove feedback."""

    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("hand-zero timeout must be positive")
    if not np.isfinite(tolerance_deg) or tolerance_deg < 0.0:
        raise ValueError("hand-zero tolerance must be non-negative")

    zero_deg = np.zeros(6, dtype=np.float64)
    expected = (0,) * 6
    gate.check("direct hand-zero command")
    receipt = hand.send(zero_deg, zero_deg)
    if receipt.targets != {"left": expected, "right": expected}:
        raise RuntimeError(
            "direct hand-zero encoding mismatch: "
            f"expected={expected} actual={receipt.targets}"
        )
    hand.wait_for_receipt(receipt, timeout=min(timeout, 1.0))
    gate.check("direct hand-zero ACK complete")
    print(
        "HAND_ZERO_COMMAND_ACK direct_unlimited targets=left:[0,0,0,0,0,0],"
        "right:[0,0,0,0,0,0]",
        flush=True,
    )

    deadline = time.monotonic() + timeout
    last_error = "no fresh hand feedback"
    while time.monotonic() < deadline:
        gate.check("direct hand-zero feedback confirmation")
        try:
            feedback = latest_control_feedback(node, max_age)
        except RuntimeError as exc:
            last_error = str(exc)
        else:
            left = np.asarray(feedback["left_hand"], dtype=np.float64)
            right = np.asarray(feedback["right_hand"], dtype=np.float64)
            left_error = float(np.max(np.abs(left)))
            right_error = float(np.max(np.abs(right)))
            if left_error <= tolerance_deg and right_error <= tolerance_deg:
                print(
                    "HAND_ZERO_FEEDBACK_CONFIRMED "
                    f"left_max_abs_deg={left_error:.3f} "
                    f"right_max_abs_deg={right_error:.3f} "
                    f"tolerance_deg={tolerance_deg:.3f}",
                    flush=True,
                )
                return feedback
            last_error = (
                f"left_max_abs_deg={left_error:.3f} "
                f"right_max_abs_deg={right_error:.3f}"
            )
        gate.wait(0.05, "direct hand-zero settling")

    raise RuntimeError(
        "timed out waiting for direct hand-zero feedback: "
        f"timeout={timeout:.3f}s tolerance_deg={tolerance_deg:.3f} "
        f"last={last_error}"
    )


class ArmCommander:
    def __init__(
        self,
        *,
        gate: ExecutionGate | None = None,
    ) -> None:
        from x2robot import connect
        from x2robot.geometry_msgs import Point, Pose, Quaternion
        from x2robot.sdk import (
            ManipulatorControlMode,
            ManipulatorControlModeParam,
            RobotModeParam,
            RobotWorkMode,
        )

        self.Point, self.Pose, self.Quaternion = Point, Pose, Quaternion
        self.ManipulatorControlMode = ManipulatorControlMode
        self.ManipulatorControlModeParam = ManipulatorControlModeParam
        self.RobotModeParam = RobotModeParam
        self.RobotWorkMode = RobotWorkMode
        self._gate = gate
        self.authority_attempted = False
        self.end_pose_enabled = False
        self.end_pose_enabled_monotonic: float | None = None
        self.initialized = False
        self._check("X2 SDK connect")
        self.robot = connect("x2://localhost:50051")
        self._check("X2 SDK connected")
        self._check("read X2 waist hold")
        self.waist = self.robot.waist.get_end_pose(timeout=2.0).pose
        self._check("read X2 waist hold complete")

    def initialize(
        self,
        read_current_anchors: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Acquire authority and capture anchors immediately before END_POSE."""

        work_mode_param = self.RobotModeParam(mode=self.RobotWorkMode.SDK)
        self._check("set X2 SDK mode")
        self.authority_attempted = True
        require_success(
            self.robot.system.set_work_mode(
                work_mode_param, timeout=5.0
            ),
            "set SDK mode",
        )
        # The body controller acknowledges set_work_mode before the internal
        # mode transition is necessarily complete.  A 0.5 s delay has proved
        # too short on a freshly restarted controller and can make the
        # following END_POSE request fail with status 2.
        self._wait(SDK_WORK_MODE_SETTLE_SECONDS, "settle X2 SDK mode")
        self._check("refresh arm anchors before END_POSE")
        left_anchor, right_anchor = read_current_anchors()
        self._check("arm anchors refreshed before END_POSE")
        end_pose_param = self.ManipulatorControlModeParam(
            mode=self.ManipulatorControlMode.MANIPULATOR_END_POSE
        )
        self._check("set X2 END_POSE mode")
        require_success(
            self.robot.robot_control.set_manipulator_control_mode(
                end_pose_param,
                timeout=5.0,
            ),
            "set END_POSE mode",
        )
        self.end_pose_enabled = True
        # Establish a complete left/right/torso target immediately after
        # END_POSE is accepted. The DWBC reasoning path requires all three.
        # These arm targets are measured poses captured before activation,
        # never model actions.
        self.send(left_anchor, right_anchor, timeout=2.0)
        self._wait(
            END_POSE_MODE_SETTLE_SECONDS,
            "settle X2 END_POSE mode",
        )
        self._check("verify X2 END_POSE mode")
        active_mode = self.robot.robot_control.get_manipulator_control_mode(
            timeout=2.0
        )
        if active_mode.mode != self.ManipulatorControlMode.MANIPULATOR_END_POSE:
            raise RuntimeError(
                "X2 END_POSE mode verification failed: "
                f"active={active_mode.mode!r}"
            )
        # Downstream feedback must be newer than the completed and verified
        # transition, not merely newer than the setter RPC acknowledgement.
        self.end_pose_enabled_monotonic = time.monotonic()
        self._check("X2 initialization complete")
        self.initialized = True

    def _check(self, stage: str) -> None:
        if self._gate is not None:
            self._gate.check(stage)

    def _wait(self, seconds: float, stage: str) -> None:
        if self._gate is None:
            time.sleep(seconds)
        else:
            self._gate.wait(seconds, stage)

    def _pose(self, pose7: np.ndarray) -> Any:
        value = np.asarray(pose7, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise RuntimeError("arm pose7 must be finite shape (7,)")
        quaternion_norm = float(np.linalg.norm(value[3:]))
        if abs(quaternion_norm - 1.0) > 1e-5:
            raise RuntimeError(f"arm quaternion is not normalized: {quaternion_norm}")
        return self.Pose(
            position=self.Point(x=value[0], y=value[1], z=value[2]),
            orientation=self.Quaternion(
                x=value[3], y=value[4], z=value[5], w=value[6]
            ),
        )

    def send(self, left: np.ndarray, right: np.ndarray, *, timeout: float) -> None:
        # Treat left/right/waist as one bounded command transaction.  Checking
        # the stop gate between sides could accept a new left target and then
        # abort before the matching right target when Ctrl+C arrives mid-RPC.
        self._check("dual arm pose command")
        require_success(
            self.robot.left_arm.set_end_pose(self._pose(left), timeout=timeout),
            "left set_end_pose",
        )
        require_success(
            self.robot.right_arm.set_end_pose(self._pose(right), timeout=timeout),
            "right set_end_pose",
        )
        require_success(
            self.robot.waist.set_end_pose(self.waist, timeout=timeout),
            "torso set_end_pose",
        )

    def hold_feedback(
        self, left: np.ndarray, right: np.ndarray, *, timeout: float
    ) -> None:
        """Best-effort soft stop; deliberately bypasses the closed execution gate."""

        if self._gate is not None and self._gate.force_emergency:
            raise RuntimeError("second stop signal interrupted arm soft hold")
        require_success(
            self.robot.left_arm.set_end_pose(self._pose(left), timeout=timeout),
            "left stop hold",
        )
        if self._gate is not None and self._gate.force_emergency:
            raise RuntimeError("second stop signal interrupted arm soft hold")
        require_success(
            self.robot.right_arm.set_end_pose(self._pose(right), timeout=timeout),
            "right stop hold",
        )
        if self._gate is not None and self._gate.force_emergency:
            raise RuntimeError("second stop signal interrupted arm soft hold")
        require_success(
            self.robot.waist.set_end_pose(self.waist, timeout=timeout),
            "torso stop hold",
        )

    def emergency_stop(self, *, timeout: float) -> None:
        require_success(
            self.robot.robot_control.emergency_stop(timeout=timeout),
            "SDK emergency stop",
        )


def matrix_pose7(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], matrix_to_quaternion_xyzw(transform[:3, :3])]
    )


def log_step(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.110.9")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--rate",
        type=positive_control_rate_arg,
        default=DEFAULT_CONTROL_RATE_HZ,
        help="any positive finite action-waypoint playback rate in Hz",
    )
    parser.add_argument(
        "--arm-command-rate",
        type=positive_control_rate_arg,
        default=DEFAULT_ARM_COMMAND_RATE_HZ,
        help=(
            "dual-arm interpolated SDK command rate; action/chunk playback "
            "continues at --rate"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_LIVE_CHUNK_SIZE,
        help="planned asynchronous switch position in the execution schedule",
    )
    parser.add_argument(
        "--inference-lead-steps",
        type=int,
        default=DEFAULT_INFERENCE_LEAD_STEPS,
        help="launch the next inference this many points before chunk-size",
    )
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument(
        "--camera-sync-max-skew-ms",
        type=float,
        default=DEFAULT_CAMERA_SYNC_MAX_SKEW_MS,
        help="maximum span across all three camera timestamps",
    )
    parser.add_argument("--rpc-timeout", type=float, default=0.2)
    parser.add_argument(
        "--v2-to-current-action-basis-config",
        type=Path,
        default=Path(__file__).with_name("V2_TO_CURRENT_ACTION_BASIS.json"),
        help="A=R_v2^T R_current matrices for output-only v2-to-current mapping",
    )
    parser.add_argument("--enable-file", type=Path, default=Path("/home/xr/pi05_umi_inference/LIVE_ENABLE"))
    parser.add_argument(
        "--arm-authority-file",
        type=Path,
        default=Path("/home/xr/pi05_umi_inference/ARM_AUTHORITY_ACTIVE"),
        help="sentinel retained until an arm hold or emergency stop is confirmed",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("/home/xr/pi05_umi_inference/live_logs"))
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until stopped")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if not 1 <= args.inference_lead_steps < args.chunk_size:
        parser.error(
            "--inference-lead-steps must satisfy 1 <= lead < chunk-size"
        )
    if (
        not np.isfinite(args.camera_sync_max_skew_ms)
        or args.camera_sync_max_skew_ms <= 0.0
    ):
        parser.error("--camera-sync-max-skew-ms must be positive")
    if args.camera_sync_max_skew_ms > 50.0:
        parser.error("live --camera-sync-max-skew-ms may not exceed 50")
    if not args.enable_file.is_file():
        parser.error(f"execution enable file is missing: {args.enable_file}")
    if args.arm_authority_file.exists():
        parser.error(
            "stale arm-authority sentinel requires external stop recovery: "
            f"{args.arm_authority_file}"
        )

    stop = threading.Event()
    gate = ExecutionGate(
        stop,
        enable_file=args.enable_file,
        execute=True,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"pi05-live-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    rclpy.init()
    node = ObservationNode(camera_sync_max_skew_ms=args.camera_sync_max_skew_ms)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="pi05-ros", daemon=True)
    spin_thread.start()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-infer")
    hand: HandCommander | None = None
    arm: ArmCommander | None = None
    pending: Future[PolicyChunk] | None = None
    worker: InferenceWorker | None = None
    interrupted = False
    failure: Exception | None = None
    arm_authority_owned = False
    client_run_id = (
        f"xr-{time.strftime('%Y%m%dT%H%M%S')}-pid{os.getpid()}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    try:
        # The first signal closes the execution gate. Interruptible waits and
        # hardware-boundary checks then unwind at a defined safe boundary; the
        # shell wrapper provides a kill watchdog for an uncooperative native
        # or network call.
        signal.signal(signal.SIGINT, gate.handle_signal)
        signal.signal(signal.SIGTERM, gate.handle_signal)
        worker = InferenceWorker(
            node,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            max_age=args.max_age,
            control_rate=args.rate,
            chunk_size=args.chunk_size,
            client_run_id=client_run_id,
            camera_sync_max_skew_ms=args.camera_sync_max_skew_ms,
            gate=gate,
            action_basis_config=args.v2_to_current_action_basis_config,
        )
        print(f"INFERENCE_RUN_ID={client_run_id}", flush=True)
        print("ACTION_MAPPING=auto_from_reviewed_checkpoint_profile", flush=True)
        print("WARMUP_INFERENCE waiting; no commands are enabled", flush=True)
        initial_future = pool.submit(worker, "pre_control_warmup")
        active = wait_future_interruptibly(
            initial_future,
            gate,
            "initial policy inference",
        )
        action_horizon = int(active.actions.shape[0])
        execution_indices = tuple(active.execution_indices)
        executable_horizon = len(execution_indices)
        policy_rate_hz = float(active.policy_rate_hz)
        async_plan = plan_chunk_prefix(
            execution_indices=execution_indices,
            chunk_size=args.chunk_size,
            lead_steps=args.inference_lead_steps,
            start_position=0,
        )
        prefetch_position = async_plan.prefetch_position
        planned_boundary_position = async_plan.planned_boundary_position
        execution_stride = (
            1
            if len(execution_indices) < 2
            else execution_indices[1] - execution_indices[0]
        )
        execution_first_index = execution_indices[0]
        print(
            f"INITIAL_CHUNK_READY seq={active.sequence} policy_ms={active.policy_ms} "
            f"server_ms={active.server_ms} execute=True "
            f"chunk_size={args.chunk_size}/{executable_horizon} "
            f"inference_lead_steps={args.inference_lead_steps} "
            f"prefetch_position={prefetch_position} "
            f"raw_horizon={action_horizon} "
            f"record={active.recording.get('relative_path')} "
            f"camera_sync_ms={active.camera_sync.get('max_pairwise_skew_ms')} "
            f"action_output_basis={active.action_output_basis.get('mode') if active.action_output_basis else 'identity'} "
            f"policy_rate_hz={policy_rate_hz:g} "
            f"action_rate_hz={args.rate:g} "
            f"arm_command_rate_hz={args.arm_command_rate:g} "
            f"execution_stride={execution_stride} limits=disabled",
            flush=True,
        )

        gate.check("pre-control observation")
        current = wait_snapshot(
            node,
            args.timeout,
            args.max_age,
            stop_check=gate.check,
        )
        gate.check("Revo2 commander construction")
        hand = HandCommander(node, gate=gate)
        hand.wait_for_bridge(args.timeout)
        # After the no-command policy warm-up succeeds, send exactly one direct
        # all-zero target to both hands. This startup action intentionally does
        # not use the live per-step hand limiter or any intermediate targets.
        current = zero_hands_direct_before_live(
            hand,
            node,
            gate,
            max_age=args.max_age,
            timeout=args.timeout,
        )
        gate.check("pre-arm Revo2 zero handshake complete")
        print(
            "REVO2_ACK_HANDSHAKE_READY stage=pre_arm_zeroed sides=left,right",
            flush=True,
        )
        gate.check("arm commander construction")
        arm = ArmCommander(gate=gate)
        gate.check("arm commander initialization")
        args.arm_authority_file.parent.mkdir(parents=True, exist_ok=True)
        with args.arm_authority_file.open("x", encoding="utf-8") as marker:
            marker.write(f"pid={os.getpid()} created_wall_time_ns={time.time_ns()}\n")
        arm_authority_owned = True
        def read_current_arm_anchors() -> tuple[np.ndarray, np.ndarray]:
            refreshed = latest_control_feedback(node, args.max_age)
            return (
                matrix_pose7(refreshed["left_eef"]),
                matrix_pose7(refreshed["right_eef"]),
            )

        arm.initialize(read_current_arm_anchors)
        if arm.end_pose_enabled_monotonic is None:
            raise RuntimeError("END_POSE mode has no completion timestamp")
        post_mode_feedback = wait_control_feedback_after(
            node,
            after_monotonic=arm.end_pose_enabled_monotonic,
            timeout=args.timeout,
            max_age=args.max_age,
            gate=gate,
            stage="post-END_POSE feedback",
        )
        # Startup has no prior live command. Use the first feedback proven newer
        # than END_POSE activation as the command refreshed during inference.
        last_command_left = matrix_pose7(post_mode_feedback["left_eef"])
        last_command_right = matrix_pose7(post_mode_feedback["right_eef"])
        arm.send(
            last_command_left,
            last_command_right,
            timeout=args.rpc_timeout,
        )
        last_arm_dispatch_started = time.monotonic()

        # Request one fresh chunk and execute it directly. There is no arm
        # settling, tracking-error, displacement, rotation or anchor-drift gate.
        pending = pool.submit(worker, "fresh_live_initial")
        active = wait_future_with_command_refresh(
            pending,
            arm=arm,
            gate=gate,
            left_hold_pose7=last_command_left,
            right_hold_pose7=last_command_right,
            rpc_timeout=args.rpc_timeout,
            rate_hz=args.rate,
            stage="fresh live policy inference",
        )
        pending = None
        # The helper may have refreshed the hold immediately before the policy
        # Future completed. Use a conservative fresh timestamp so the first
        # interpolation command cannot follow that hold too closely.
        last_arm_dispatch_started = time.monotonic()
        gate.check("fresh live chunk format validation")
        assert_executable_chunk_prefix(
            active,
            args.chunk_size,
            action_horizon=action_horizon,
            execution_indices=execution_indices,
        )
        # Capture feedback newer than the final inference-wait hold and use it
        # as the exact interpolation start.  The per-chunk overlay below is
        # applied to a decoded model endpoint, never as a synthetic waypoint.
        prelive_feedback = wait_control_feedback_after(
            node,
            after_monotonic=last_arm_dispatch_started,
            timeout=args.timeout,
            max_age=args.max_age,
            gate=gate,
            stage="pre-live execution feedback",
        )
        last_command_left = matrix_pose7(prelive_feedback["left_eef"])
        last_command_right = matrix_pose7(prelive_feedback["right_eef"])
        chunk_left_z_execution_anchor = ChunkLeftZExecutionAnchor(
            CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M
        )
        print(
            "CHUNK_LEFT_Z_EXECUTION_ANCHOR_CONFIG "
            "frame=base_link axis=z "
            f"offset_m={CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M} "
            "scope=all_executed_actions_in_each_chunk "
            "delta_eef=unchanged no_second_action_recovery=true",
            flush=True,
        )

        gate.check("pre-live feedback")
        prelive_handshake = hand.send(
            prelive_feedback["left_hand"], prelive_feedback["right_hand"]
        )
        hand.wait_for_receipt(prelive_handshake, timeout=0.6)
        last_hand_output_monotonic = time.monotonic()
        gate.check("pre-live Revo2 handshake complete")
        print("REVO2_ACK_HANDSHAKE_READY stage=pre_live sides=left,right", flush=True)
        gate.mark_live_output_enabled()
        print(
            "LIVE_OUTPUT_ENABLED arms+hands chunked=async_double_buffer "
            f"chunk_size={args.chunk_size}/{executable_horizon} "
            f"lead_steps={args.inference_lead_steps} "
            f"prefetch_position={prefetch_position} "
            f"planned_boundary={planned_boundary_position} "
            f"raw_horizon={action_horizon} "
            f"policy_rate_hz={policy_rate_hz:g} "
            f"action_rate_hz={args.rate:g} "
            f"arm_command_rate_hz={args.arm_command_rate:g} "
            "arm_interpolation=right_relative_pose7 "
            f"chunk_left_base_z_execution_anchor_offset_m="
            f"{CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M} "
            f"execution_stride={execution_stride} "
            f"fresh_seq={active.sequence} "
            f"record={active.recording.get('relative_path')} "
            f"camera_sync_ms={active.camera_sync.get('max_pairwise_skew_ms')} "
            f"action_output_basis={active.action_output_basis.get('mode') if active.action_output_basis else 'identity'}",
            flush=True,
        )

        step = 0
        # The robot was held at the fresh_live_initial observation anchor, so
        # its first chunk intentionally begins at action 0.  Latency skipping is
        # applied only to replacements captured while the old chunk kept moving.
        buffers = AsyncChunkBuffer(
            active=active,
            chunk_size=args.chunk_size,
            lead_steps=args.inference_lead_steps,
            executable_horizon=executable_horizon,
        )
        deadline = time.monotonic()
        with log_path.open("a", encoding="utf-8") as stream:
            while not args.max_steps or step < args.max_steps:
                gate.check("control loop")
                now = time.monotonic()
                if now < deadline:
                    gate.wait_until(
                        deadline, f"{args.rate:g}Hz control deadline"
                    )
                gate.check("control loop deadline")
                # One decoded model target remains one action-rate waypoint.
                # Arm interpolation fills this interval without advancing the
                # chunk cursor or changing the model trajectory duration.
                segment_end_deadline = deadline + 1.0 / args.rate

                # The inference worker owns the only WebSocket connection.  Poll
                # its Future without blocking the 10 Hz command loop, then keep
                # the complete chunk and its request-owned anchors together in
                # the standby buffer until the planned switch position.
                if pending is not None and pending.done():
                    candidate = pending.result()
                    pending = None
                    gate.check("asynchronous replacement format validation")
                    assert_executable_chunk_prefix(
                        candidate,
                        args.chunk_size,
                        action_horizon=action_horizon,
                        execution_indices=execution_indices,
                    )
                    buffers.accept_ready(
                        candidate, ready_monotonic_ns=time.monotonic_ns()
                    )
                    print(
                        f"STANDBY_READY seq={candidate.sequence} "
                        f"active_seq={buffers.active.sequence} "
                        f"active_cursor={buffers.cursor} "
                        f"policy_ms={candidate.policy_ms} "
                        f"server_ms={candidate.server_ms} "
                        f"record={candidate.recording.get('relative_path')}",
                        flush=True,
                    )

                # A ready replacement is switched atomically only at/after the
                # requested chunk boundary.  Its first action is chosen from the
                # request's XR-local observation time; action 0 is never replayed
                # after it has already expired while the old chunk kept moving.
                if buffers.at_or_after_boundary and buffers.standby is not None:
                    # The selected candidate target is reached at the end of
                    # this interpolated segment, not at its start.
                    switch_deadline = max(
                        segment_end_deadline, time.monotonic()
                    )
                    switch_deadline_ns = int(round(switch_deadline * 1e9))
                    switch = buffers.try_switch(
                        next_send_deadline_ns=switch_deadline_ns,
                        control_rate_hz=args.rate,
                    )
                    observation_age_ms = (
                        None
                        if switch.observation_age_ns is None
                        else switch.observation_age_ns / 1e6
                    )
                    standby_wait_ms = (
                        None
                        if switch.standby_wait_ns is None
                        else switch.standby_wait_ns / 1e6
                    )
                    if switch.stale:
                        print(
                            "STALE_CANDIDATE_DISCARDED "
                            f"seq={switch.candidate_sequence} "
                            f"age_ms={observation_age_ms:.1f} "
                            f"standby_wait_ms={standby_wait_ms} "
                            f"selected_position="
                            f"{None if switch.selection is None else switch.selection.execution_position} "
                            f"planned_boundary={planned_boundary_position} "
                            f"timing_rate_hz={switch.timing_rate_hz:g} "
                            "action=request_fresh_replacement",
                            flush=True,
                        )
                    elif switch.swapped:
                        selection = switch.selection
                        assert selection is not None
                        active = buffers.active
                        print(
                            f"CHUNK_SWAP old_seq={switch.old_sequence} "
                            f"seq={active.sequence} "
                            f"planned_boundary={planned_boundary_position} "
                            f"actual_old_cursor={switch.old_cursor} "
                            f"old_tail_steps_used={switch.old_tail_steps_used} "
                            f"observation_age_ms={observation_age_ms:.1f} "
                            f"standby_wait_ms={standby_wait_ms} "
                            f"k_start_position={selection.execution_position} "
                            f"k_start_raw={selection.raw_policy_index} "
                            f"timing_rate_hz={switch.timing_rate_hz:g} "
                            f"policy_ms={active.policy_ms} server_ms={active.server_ms} "
                            f"record={active.recording.get('relative_path')} "
                            f"camera_sync_ms={active.camera_sync.get('max_pairwise_skew_ms')} "
                            f"transport_age_ms="
                            f"{active.camera_sync.get('observation_transport_age_ms')}",
                            flush=True,
                        )

                # Start exactly one replacement request lead_steps before the
                # planned boundary.  If a latency-compensated chunk starts near
                # that boundary this condition deliberately launches at once.
                if buffers.should_launch(pending_exists=pending is not None):
                    gate.check("asynchronous inference prefetch launch")
                    pending = pool.submit(worker, "async_chunk_replacement")
                    buffers.mark_request_launched()
                    print(
                        f"INFERENCE_PREFETCH active_seq={buffers.active.sequence} "
                        f"trigger_cursor={buffers.cursor} "
                        f"prefetch_position={prefetch_position} "
                        f"planned_boundary={planned_boundary_position}",
                        flush=True,
                    )

                # The raw policy horizon after chunk_size is a bridge, not dead
                # data.  Only when all executable rows are consumed do we hold
                # the last complete target while waiting for the single Future.
                if buffers.exhausted:
                    assert arm is not None and hand is not None
                    if pending is None:
                        gate.check("exhausted-horizon inference launch")
                        pending = pool.submit(worker, "async_horizon_recovery")
                        buffers.mark_request_launched()
                    print(
                        "ASYNC_BUFFER_EXHAUSTED "
                        f"active_seq={buffers.active.sequence} "
                        f"cursor={buffers.cursor} "
                        "action=refresh_last_command_until_result",
                        flush=True,
                    )
                    candidate = wait_future_with_command_refresh(
                        pending,
                        arm=arm,
                        gate=gate,
                        left_hold_pose7=last_command_left,
                        right_hold_pose7=last_command_right,
                        rpc_timeout=args.rpc_timeout,
                        rate_hz=args.rate,
                        stream=stream,
                        stage="asynchronous buffer exhaustion",
                    )
                    pending = None
                    # The refresh helper may have sent a hold immediately
                    # before returning. Start the next interpolation no sooner
                    # than one full arm-command period from this conservative
                    # timestamp.
                    last_arm_dispatch_started = time.monotonic()
                    gate.check("exhausted-horizon replacement validation")
                    assert_executable_chunk_prefix(
                        candidate,
                        args.chunk_size,
                        action_horizon=action_horizon,
                        execution_indices=execution_indices,
                    )
                    # The blocking arm-hold fallback can outlive the Revo2 ACK.
                    # Refresh only here, before k_start is computed, so the ACK
                    # wait is included in replacement latency compensation.
                    if time.monotonic() - last_hand_output_monotonic > 0.5:
                        gate.check("exhausted-horizon hand hold refresh")
                        hand_feedback = latest_hand_feedback(node, args.max_age)
                        hand_hold_receipt = hand.send(
                            hand_feedback["left_hand"],
                            hand_feedback["right_hand"],
                        )
                        hand.wait_for_receipt(hand_hold_receipt, timeout=0.6)
                        last_hand_output_monotonic = time.monotonic()
                    buffers.accept_ready(
                        candidate, ready_monotonic_ns=time.monotonic_ns()
                    )
                    # A blocking fallback disrupts the periodic phase.  Realign
                    # once here; normal asynchronous swaps never reset deadline.
                    deadline = time.monotonic()
                    continue

                if buffers.in_old_tail:
                    print(
                        f"OLD_TAIL_CONTINUE seq={buffers.active.sequence} "
                        f"cursor={buffers.cursor} pending={pending is not None} "
                        f"standby={buffers.standby is not None}",
                        flush=True,
                    )

                gate.check("control feedback read")
                active = buffers.active
                segment_started_at = time.monotonic()
                index = buffers.cursor
                raw_policy_index = active.execution_indices[index]
                raw_left = active.left_pose7[raw_policy_index]
                raw_right = active.right_pose7[raw_policy_index]
                raw_left_hand = active.left_hand_deg[raw_policy_index]
                raw_right_hand = active.right_hand_deg[raw_policy_index]
                # The model rows remain shared-anchor relative actions decoded
                # exactly once as T_anchor @ Delta[k].  Apply the requested
                # same base-link Z execution-anchor overlay to every target in
                # this chunk. A latency-selected replacement may begin at k>0;
                # that first executed row and all later rows receive the same
                # offset, preserving the chunk's Delta EEF differences.
                decoded_left, right, left_hand, right_hand = (
                    validated_direct_action_targets(active, raw_policy_index)
                )
                left, chunk_first_action = chunk_left_z_execution_anchor.prepare(
                    active.sequence, decoded_left
                )
                # Read fresh feedback for command-link health only. The
                # interpolation start remains the last complete target accepted
                # by the same uninterrupted END_POSE execution group.
                latest_control_feedback(node, args.max_age)
                interpolation_start_left = last_command_left.copy()
                interpolation_start_right = last_command_right.copy()
                arm_points = interpolate_arm_segment(
                    start_left_pose7=interpolation_start_left,
                    start_right_pose7=interpolation_start_right,
                    target_left_pose7=left,
                    target_right_pose7=right,
                    action_rate_hz=args.rate,
                    arm_command_rate_hz=args.arm_command_rate,
                )
                assert arm is not None and hand is not None
                gate.check("live hand command dispatch")
                resumed_after_gap = (
                    segment_started_at - last_hand_output_monotonic > 0.5
                )
                # Revo2 already executes each endpoint over 100 ms at 10 Hz.
                # Publish one hand endpoint per execution-group waypoint; only
                # the arms receive higher-frequency interpolation points.
                if resumed_after_gap:
                    receipt = hand.send(left_hand, right_hand)
                    hand.wait_for_receipt(receipt, timeout=0.6)
                else:
                    hand.assert_ack_fresh(0.6)
                    receipt = hand.send(left_hand, right_hand)
                encoded_hands = receipt.targets
                last_hand_output_monotonic = time.monotonic()

                interpolation_log: list[dict[str, Any]] = []
                sent_arm_substeps = 0
                for point_index, point in enumerate(arm_points):
                    gate.check(
                        f"arm interpolation {point.substep}/{point.substeps}"
                    )
                    due = deadline + point.offset_seconds
                    now_before_wait = time.monotonic()
                    next_due = (
                        None
                        if point_index + 1 == len(arm_points)
                        else deadline
                        + arm_points[point_index + 1].offset_seconds
                    )
                    dispatch_deadline = next_dispatch_deadline(
                        now=now_before_wait,
                        due=due,
                        next_due=next_due,
                        last_dispatch_started=last_arm_dispatch_started,
                        arm_command_rate_hz=args.arm_command_rate,
                    )
                    if dispatch_deadline is None:
                        interpolation_log.append(
                            {
                                "substep": point.substep,
                                "substeps": point.substeps,
                                "alpha": point.alpha,
                                "scheduled_offset_ms": (
                                    point.offset_seconds * 1000.0
                                ),
                                "sent": False,
                                "reason": "expired_intermediate_no_burst",
                            }
                        )
                        print(
                            "ARM_INTERPOLATION_SKIP "
                            f"step={step} chunk={active.sequence}:{index} "
                            f"substep={point.substep}/{point.substeps} "
                            "reason=expired_intermediate_no_burst",
                            flush=True,
                        )
                        continue
                    if now_before_wait < dispatch_deadline:
                        gate.wait_until(
                            dispatch_deadline,
                            "arm interpolation deadline "
                            f"{point.substep}/{point.substeps}",
                        )
                    gate.check(
                        "arm interpolation dispatch "
                        f"{point.substep}/{point.substeps}"
                    )
                    arm_sent_at = time.monotonic()
                    arm.send(
                        point.left_pose7,
                        point.right_pose7,
                        timeout=args.rpc_timeout,
                    )
                    last_arm_dispatch_started = arm_sent_at
                    # Update only after the complete dual-arm SDK call returns
                    # successfully. A stop/failure can therefore hold the last
                    # command that was actually accepted.
                    last_command_left = point.left_pose7.copy()
                    last_command_right = point.right_pose7.copy()
                    sent_arm_substeps += 1
                    interpolation_log.append(
                        {
                            "substep": point.substep,
                            "substeps": point.substeps,
                            "alpha": point.alpha,
                            "scheduled_offset_ms": (
                                point.offset_seconds * 1000.0
                            ),
                            "scheduled_lateness_ms": max(
                                0.0, (arm_sent_at - due) * 1000.0
                            ),
                            "sent": True,
                            "left_pose7": point.left_pose7.tolist(),
                            "right_pose7": point.right_pose7.tolist(),
                            "dispatch_ms": (
                                (time.monotonic() - arm_sent_at) * 1000.0
                            ),
                        }
                    )
                if not arm_points or arm_points[-1].alpha != 1.0:
                    raise RuntimeError(
                        "arm interpolation did not contain the selected endpoint"
                    )
                # Smoothing may add commands but may not change the selected
                # prefix/model endpoint. Quaternion sign may be flipped because
                # q and -q are the same rotation; choosing the previous
                # command's hemisphere avoids a numeric SDK discontinuity.
                def same_physical_pose(
                    actual: np.ndarray, expected: np.ndarray
                ) -> bool:
                    return bool(
                        np.array_equal(actual[:3], expected[:3])
                        and np.isclose(
                            abs(float(np.dot(actual[3:], expected[3:]))),
                            1.0,
                            atol=1e-12,
                        )
                    )

                if not same_physical_pose(
                    last_command_left, left
                ) or not same_physical_pose(last_command_right, right):
                    raise RuntimeError(
                        "arm interpolation failed to reach the selected execution-group pose"
                    )
                # Commit only after the alpha=1 dual-arm SDK call has returned
                # and the accepted endpoint has been verified.  A stale
                # standby chunk, Ctrl+C, or a failed SDK call therefore cannot
                # commit the new execution-anchor transition.
                if chunk_first_action:
                    chunk_left_z_execution_anchor.commit(active.sequence)
                execution_group_id = f"{client_run_id}:chunk:{active.sequence}"
                model_action30 = (
                    active.actions[raw_policy_index].astype(float).tolist()
                )
                robot_action30_current = (
                    active.actions[raw_policy_index]
                    if active.robot_actions_current is None
                    else active.robot_actions_current[raw_policy_index]
                ).astype(float).tolist()
                log_step(
                    stream,
                    {
                        "wall_time": time.time(),
                        "record_type": "model_action",
                        "sent": True,
                        "global_step": step,
                        "execution_group_id": execution_group_id,
                        "execution_group_position": index,
                        "chunk_sequence": active.sequence,
                        "chunk_index": index,
                        "raw_policy_index": raw_policy_index,
                        "client_run_id": client_run_id,
                        "inference_recording": active.recording,
                        "camera_sync": active.camera_sync,
                        "chunk_observation_wall_time_ns": (
                            active.observation_wall_time_ns
                        ),
                        "chunk_observation_monotonic_ns": (
                            active.observation_monotonic_ns
                        ),
                        "model_action30": model_action30,
                        "model_action_basis": active.action_output_basis.get(
                            "model_output_basis", "unknown"
                        ),
                        "robot_action30_current": robot_action30_current,
                        "action_output_basis": active.action_output_basis,
                        "model_input_state30": active.input_state30.astype(float).tolist(),
                        "left_chunk_anchor_T": active.left_anchor_T.astype(float).tolist(),
                        "right_chunk_anchor_T": active.right_anchor_T.astype(float).tolist(),
                        "chunk_left_z_execution_anchor": {
                            "applied": True,
                            "first_executed_action": chunk_first_action,
                            "frame": "base_link",
                            "axis": "z",
                            "configured_offset_m": (
                                CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M
                            ),
                            "applied_translation_m": [
                                0.0,
                                0.0,
                                CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M,
                            ],
                            "scope": (
                                "all_executed_actions_in_each_chunk"
                            ),
                            "delta_eef_difference_policy": "unchanged",
                            "second_action_recovery_m": 0.0,
                            "decoded_left_pose7": decoded_left.tolist(),
                            "execution_target_left_pose7": left.tolist(),
                        },
                        "raw_left_pose7": raw_left.astype(float).tolist(),
                        "raw_right_pose7": raw_right.astype(float).tolist(),
                        "interpolation_start_left_pose7": (
                            interpolation_start_left.tolist()
                        ),
                        "interpolation_start_right_pose7": (
                            interpolation_start_right.tolist()
                        ),
                        "arm_interpolation": {
                            "formula": (
                                "D=inv(T_last)@T_target; "
                                "T_s=T_last@Interp(I,D,alpha)"
                            ),
                            "relative_composition": "right",
                            "commands": interpolation_log,
                        },
                        "decoded_target_left_pose7": decoded_left.tolist(),
                        "decoded_target_right_pose7": right.tolist(),
                        "execution_target_left_pose7": left.tolist(),
                        "execution_target_right_pose7": right.tolist(),
                        "command_left_pose7": last_command_left.tolist(),
                        "command_right_pose7": last_command_right.tolist(),
                        "raw_left_hand_deg": raw_left_hand.astype(float).tolist(),
                        "raw_right_hand_deg": raw_right_hand.astype(float).tolist(),
                        "command_left_hand_deg": left_hand.tolist(),
                        "command_right_hand_deg": right_hand.tolist(),
                        "left_hand_motor": list(encoded_hands["left"]),
                        "right_hand_motor": list(encoded_hands["right"]),
                        "control": {
                            "policy_rate_hz": policy_rate_hz,
                            "action_rate_hz": args.rate,
                            "arm_command_rate_hz": args.arm_command_rate,
                            "arm_substeps_planned": len(arm_points),
                            "arm_substeps_sent": sent_arm_substeps,
                            "raw_action_horizon": action_horizon,
                            "execution_stride": execution_stride,
                            "execution_first_index": execution_first_index,
                            "executable_horizon": executable_horizon,
                            "async_double_buffer": True,
                            "inference_lead_steps": args.inference_lead_steps,
                            "prefetch_position": prefetch_position,
                            "planned_boundary_position": (
                                planned_boundary_position
                            ),
                            "pending_inference": pending is not None,
                            "standby_ready": buffers.standby is not None,
                            "chunk_first_action": chunk_first_action,
                            "chunk_left_base_z_execution_anchor_offset_m": (
                                CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M
                            ),
                            "limits_enabled": False,
                        },
                    },
                )
                elapsed = time.monotonic() - segment_started_at
                if chunk_first_action:
                    print(
                        "CHUNK_LEFT_Z_EXECUTION_ANCHOR_STARTED "
                        f"group={execution_group_id} "
                        f"chunk={active.sequence} "
                        f"chunk_index={index} "
                        f"raw_policy_index={raw_policy_index} "
                        "frame=base_link axis=z "
                        f"offset_m={CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M} "
                        f"decoded_z_m={decoded_left[2]} "
                        f"execution_z_m={left[2]}",
                        flush=True,
                    )
                print(
                    f"STEP {step} chunk={active.sequence}:{index} sent=True "
                    f"group={execution_group_id} "
                    f"raw_policy_index={raw_policy_index} "
                    f"chunk_first_action={chunk_first_action} "
                    f"left_base_z_offset_m="
                    f"{CHUNK_LEFT_BASE_Z_EXECUTION_ANCHOR_OFFSET_M} "
                    f"action_rate_hz={args.rate:g} "
                    f"arm_command_rate_hz={args.arm_command_rate:g} "
                    f"arm_substeps={sent_arm_substeps}/{len(arm_points)} "
                    f"raw_horizon={action_horizon} "
                    "limited=False "
                    f"segment_ms={elapsed * 1000.0:.1f}",
                    flush=True,
                )
                step += 1
                # One cursor advance per original model waypoint, regardless
                # of how many SDK commands were added inside the interval.
                buffers.advance_after_send()
                now_after_send = time.monotonic()
                interpolation_period = 1.0 / args.arm_command_rate
                lateness = max(0.0, now_after_send - segment_end_deadline)
                if lateness >= interpolation_period:
                    # Do not issue catch-up bursts. A seriously late SDK call
                    # starts a fresh full-duration action segment.
                    deadline = now_after_send
                    print(
                        "ARM_INTERPOLATION_OVERRUN realigning "
                        f"action_rate_hz={args.rate:g} "
                        f"arm_command_rate_hz={args.arm_command_rate:g} "
                        f"late_ms={lateness * 1000.0:.1f}",
                        flush=True,
                    )
                else:
                    deadline = segment_end_deadline
    except StopRequested as exc:
        interrupted = True
        print(f"STOP_REQUESTED reason={exc}", flush=True)
    except Exception as exc:
        failure = exc
        raise
    finally:
        should_hold = arm is not None and arm.authority_attempted
        gate.request_stop("controller shutdown")
        if worker is not None:
            worker.abort_io()
        if pending is not None:
            pending.cancel()

        stop_action = "not_required"
        if not should_hold and hand is not None and hand.has_published_target:
            try:
                hand_only_feedback = latest_hand_feedback(node, args.max_age)
                hand.hold_feedback(
                    hand_only_feedback["left_hand"],
                    hand_only_feedback["right_hand"],
                    timeout=0.25,
                )
                print(
                    "STOP_ACTION REVO2_SOFT_HOLD_CONFIRMED "
                    "source=current_feedback_before_arm_authority",
                    flush=True,
                )
            except Exception as hand_hold_exc:
                # The shell removes LIVE_ENABLE and stops the absolute bridge
                # immediately after this process exits. This is best-effort for
                # a stop/failure that occurs during startup hand zeroing.
                print(
                    "STOP_ACTION REVO2_SOFT_HOLD_FAILED "
                    "before_arm_authority "
                    f"error={type(hand_hold_exc).__name__}:{hand_hold_exc}",
                    flush=True,
                )
        if should_hold:
            if gate.force_emergency or not arm.end_pose_enabled:
                try:
                    arm.emergency_stop(timeout=0.5)
                    stop_action = (
                        "sdk_emergency_stop"
                        if gate.force_emergency
                        else "sdk_emergency_stop_during_partial_initialization"
                    )
                    print("STOP_ACTION SDK_EMERGENCY_STOP_CONFIRMED", flush=True)
                except Exception as emergency_exc:
                    stop_action = "sdk_emergency_stop_failed"
                    print(
                        "CRITICAL STOP_ACTION SDK_EMERGENCY_STOP_FAILED "
                        f"error={type(emergency_exc).__name__}:{emergency_exc}",
                        flush=True,
                    )
            else:
                try:
                    stop_feedback = latest_control_feedback(node, args.max_age)
                    # Stop arm motion first. Hand feedback/ACK may take a full
                    # bridge period and must not delay the dual-arm hold.
                    arm.hold_feedback(
                        matrix_pose7(stop_feedback["left_eef"]),
                        matrix_pose7(stop_feedback["right_eef"]),
                        timeout=0.25,
                    )
                    stop_action = "soft_hold_current_feedback"
                    print(
                        "STOP_ACTION SOFT_HOLD_CONFIRMED source=current_feedback",
                        flush=True,
                    )
                    if hand is not None:
                        try:
                            hand.hold_feedback(
                                stop_feedback["left_hand"],
                                stop_feedback["right_hand"],
                                timeout=0.25,
                            )
                            print(
                                "STOP_ACTION REVO2_SOFT_HOLD_CONFIRMED "
                                "source=current_feedback",
                                flush=True,
                            )
                        except Exception as hand_hold_exc:
                            # The shell stops the absolute bridge immediately
                            # after this process exits. Keep the confirmed arm
                            # hold even if the hand ACK path is unavailable.
                            print(
                                "STOP_ACTION REVO2_SOFT_HOLD_FAILED "
                                f"error={type(hand_hold_exc).__name__}:"
                                f"{hand_hold_exc}",
                                flush=True,
                            )
                            if gate.force_emergency:
                                raise
                except Exception as hold_exc:
                    print(
                        "STOP_ACTION SOFT_HOLD_FAILED; escalating to SDK emergency stop "
                        f"error={type(hold_exc).__name__}:{hold_exc}",
                        flush=True,
                    )
                    try:
                        arm.emergency_stop(timeout=0.5)
                        stop_action = "sdk_emergency_stop_after_hold_failure"
                        print("STOP_ACTION SDK_EMERGENCY_STOP_CONFIRMED", flush=True)
                    except Exception as emergency_exc:
                        stop_action = "all_stop_actions_failed"
                        print(
                            "CRITICAL STOP_ACTION SDK_EMERGENCY_STOP_FAILED "
                            f"error={type(emergency_exc).__name__}:{emergency_exc}",
                            flush=True,
                        )
                if gate.force_emergency and stop_action == "soft_hold_current_feedback":
                    try:
                        arm.emergency_stop(timeout=0.5)
                        stop_action = "sdk_emergency_stop_after_second_signal"
                        print(
                            "STOP_ACTION SECOND_SIGNAL_SDK_EMERGENCY_STOP_CONFIRMED",
                            flush=True,
                        )
                    except Exception as emergency_exc:
                        stop_action = "second_signal_emergency_stop_failed"
                        print(
                            "CRITICAL STOP_ACTION SECOND_SIGNAL_SDK_EMERGENCY_STOP_FAILED "
                            f"error={type(emergency_exc).__name__}:{emergency_exc}",
                            flush=True,
                        )

        stop_confirmed = (
            arm is None
            or not arm.authority_attempted
            or stop_action
            in {
                "soft_hold_current_feedback",
                "sdk_emergency_stop",
                "sdk_emergency_stop_during_partial_initialization",
                "sdk_emergency_stop_after_hold_failure",
                "sdk_emergency_stop_after_second_signal",
            }
        )
        if arm_authority_owned and stop_confirmed:
            try:
                args.arm_authority_file.unlink(missing_ok=True)
                arm_authority_owned = False
                print("ARM_AUTHORITY_RELEASE_CONFIRMED", flush=True)
            except OSError as marker_exc:
                print(
                    "CRITICAL ARM_AUTHORITY_SENTINEL_CLEAR_FAILED "
                    f"error={type(marker_exc).__name__}:{marker_exc}",
                    flush=True,
                )
        elif arm_authority_owned:
            print(
                "CRITICAL ARM_AUTHORITY_ACTIVE stop action was not confirmed; "
                f"sentinel={args.arm_authority_file}",
                flush=True,
            )

        # Closing the WebSocket above releases an in-flight infer. Join the
        # worker before tearing down its ROS observation source so no worker
        # can race a destroyed node.
        pool.shutdown(wait=True, cancel_futures=True)
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(
        f"STOPPED publishing ceased stop_action={stop_action} log={log_path}",
        flush=True,
    )
    if interrupted and gate.signal_number is not None:
        return 128 + gate.signal_number
    if failure is not None:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
