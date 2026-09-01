#!/usr/bin/env python3
"""Read-only XR preflight for a checkpoint-selected UMI policy profile."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from camera_sync import DEFAULT_STATE_SYNC_MAX_SKEW_MS
from low_latency_policy_client import LowLatencyWebsocketClientPolicy
from umi_live_contract import (
    GENERIC_CONTRACT_SOURCE,
    GENERIC_PROMPT_SOURCE,
    validate_policy_contract_metadata,
    validated_execution_schedule,
    validated_head_camera_preprocess,
)


def validated_contract_action_shape(
    contract: Mapping[str, Any],
) -> tuple[int, int]:
    """Return the reviewed action shape, rejecting ambiguous JSON values."""

    values: list[int] = []
    for key in ("action_horizon", "action_dim"):
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"policy contract {key} must be a positive integer; got {value!r}"
            )
        values.append(value)
    return values[0], values[1]


def has_explicit_execution_schedule(contract: Mapping[str, Any]) -> bool:
    return any(
        key in contract
        for key in (
            "execution_rate_hz",
            "execution_stride",
            "execution_first_index",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--v2-to-current-action-basis-config",
        type=Path,
        default=Path(__file__).with_name("V2_TO_CURRENT_ACTION_BASIS.json"),
    )
    parser.add_argument("--requested-control-rate", type=float, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    args = parser.parse_args()

    policy = LowLatencyWebsocketClientPolicy(host=args.host, port=args.port)
    try:
        metadata = policy.get_server_metadata()
        contract = validate_policy_contract_metadata(metadata)
        action_horizon, action_dim = validated_contract_action_shape(contract)
        schedule = validated_execution_schedule(contract)
        schedule_explicit = has_explicit_execution_schedule(contract)
        head_preprocess = validated_head_camera_preprocess(contract)
        if (
            not math.isfinite(args.requested_control_rate)
            or args.requested_control_rate <= 0.0
        ):
            raise RuntimeError("requested control rate must be positive")
        if (
            schedule_explicit
            and not math.isclose(
                args.requested_control_rate,
                float(schedule["execution_rate_hz"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "explicit policy execution schedule requires the reviewed "
                f"{schedule['execution_rate_hz']:g} Hz robot rate; "
                f"requested={args.requested_control_rate:g} Hz"
            )
        if not 1 <= args.chunk_size <= int(schedule["executable_horizon"]):
            raise RuntimeError(
                "chunk size is outside the executable horizon: "
                f"chunk_size={args.chunk_size} executable_horizon="
                f"{schedule['executable_horizon']} raw_horizon={action_horizon}"
            )
        expected_prompt_source = (
            GENERIC_PROMPT_SOURCE
            if contract.get("contract_source") == GENERIC_CONTRACT_SOURCE
            else "checkpoint_profile"
        )
        if metadata.get("prompt_source") != expected_prompt_source:
            raise RuntimeError(
                "policy server does not enforce the deployment task prompt"
            )
        if metadata.get("e6_eye") != contract["cam_high_eye"]:
            raise RuntimeError(
                "E6 eye does not match the selected profile: "
                f"server={metadata.get('e6_eye')!r} "
                f"profile={contract['cam_high_eye']!r}"
            )
        if metadata.get("recording_enabled") is not True:
            raise RuntimeError("live profile requires server recording")
        if metadata.get("e6_frame_live") is not True:
            raise RuntimeError("live profile requires a verified live E6 frame")
        if metadata.get("live_actuation_allowed") is not True:
            raise RuntimeError(
                "policy server did not complete the live GPU warm-up"
            )
        if metadata.get("camera_sync_required") is not True:
            raise RuntimeError("live profile requires three-camera synchronization")
        server_sync_limit = float(
            metadata.get("camera_sync_max_skew_ms", float("nan"))
        )
        if (
            not math.isfinite(server_sync_limit)
            or server_sync_limit <= 0.0
            or server_sync_limit > 50.0
        ):
            raise RuntimeError(
                "policy server camera sync limit exceeds live safety contract: "
                f"server={server_sync_limit}ms maximum=50ms"
            )
        server_state_limit = float(
            metadata.get("state_sync_max_skew_ms", float("nan"))
        )
        if (
            not math.isfinite(server_state_limit)
            or server_state_limit <= 0.0
            or server_state_limit > DEFAULT_STATE_SYNC_MAX_SKEW_MS
        ):
            raise RuntimeError(
                "policy server state sync limit exceeds live safety contract: "
                f"server={server_state_limit}ms "
                f"maximum={DEFAULT_STATE_SYNC_MAX_SKEW_MS}ms"
            )
        if contract["action_output_basis"] == "v2_to_current":
            actual_sha = hashlib.sha256(
                args.v2_to_current_action_basis_config.read_bytes()
            ).hexdigest()
            if actual_sha != contract["action_basis_config_sha256"]:
                raise RuntimeError(
                    "local v2_to_current mapping hash mismatch: "
                    f"expected={contract['action_basis_config_sha256']} "
                    f"actual={actual_sha}"
                )
        receipt = (
            "PROFILE_PREFLIGHT_OK "
            f"task={contract['task_id']} profile={contract['profile_id']} "
            f"checkpoint={metadata['checkpoint']} asset_id={metadata['asset_id']} "
            f"action_shape=({action_horizon},{action_dim}) "
            f"policy_rate_hz={schedule['policy_rate_hz']:g} "
            f"execution_rate_hz={schedule['execution_rate_hz']:g} "
            f"execution_stride={schedule['stride']} "
            f"execution_first_index={schedule['first_index']} "
            f"executable_horizon={schedule['executable_horizon']} "
            f"schedule_explicit={schedule_explicit} "
            f"cam_high_mode={metadata['cam_high_mode_effective']} "
            f"cam_high_preprocess={head_preprocess['mode']} "
            f"mapping={contract['action_output_basis']}+"
            f"{contract['eef_anchor']}+{contract['se3_composition']} "
            f"server_session={metadata.get('server_session_id')}"
        )
        print(receipt)
    finally:
        connection = getattr(policy, "_ws", None)
        close_socket = getattr(connection, "close_socket", None)
        if callable(close_socket):
            close_socket()


if __name__ == "__main__":
    main()
