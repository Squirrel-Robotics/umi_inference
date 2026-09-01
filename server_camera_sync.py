#!/usr/bin/env python3
"""Policy-server helpers for validating and completing camera sync proofs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from camera_sync import (
    CLOCK_DOMAIN,
    CLOCK_SYNC_SCHEMA,
    DEFAULT_OBSERVATION_INTERVAL_MS,
    DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS,
    DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    OBSERVATION_TRANSPORT_AGE_POLICY,
    SYNC_SCHEMA,
    validate_observation_interval,
    validate_state_alignment,
)


class ClockSyncUncertainForAlignment(RuntimeError):
    """A valid clock proof whose uncertainty requires a fresh observation."""

    def __init__(self, uncertainty_ms: float, max_uncertainty_ms: float) -> None:
        self.uncertainty_ms = float(uncertainty_ms)
        self.max_uncertainty_ms = float(max_uncertainty_ms)
        super().__init__(
            "XR/5090 clock proof is too uncertain for camera alignment: "
            f"uncertainty={self.uncertainty_ms:.3f}ms "
            f"limit={self.max_uncertainty_ms:.3f}ms"
        )


@dataclass(frozen=True)
class ParsedClientSync:
    """Server-validated view of the XR camera/state synchronization proof.

    The original dictionaries are retained so the response and recorder keep
    the existing wire fields exactly.  Derived values are computed once here
    instead of being reconstructed independently by each policy wrapper.
    """

    raw: dict[str, Any]
    clock_sync: dict[str, Any]
    cameras: dict[str, Any]
    reference_wall_times_ns: dict[str, int]
    observation_wall_time_ns: int
    state_first_observation_wall_time_ns: int
    state_observation_interval_ms: float
    state_alignment: dict[str, Any]
    server_receive_wall_time_ns: int
    clock_uncertainty_ms: float
    clock_proof_age_ms: float
    max_sync_skew_ms: float
    effective_sync_limit_ms: float
    observation_transport_age_ms: float
    worst_observation_transport_age_ms: float
    client_send_wall_time_ns: Any
    source_to_send_ms: float | None
    send_to_receive_ms: float | None


def select_best_timestamp(
    candidate_times_ns: Sequence[int],
    reference_times_ns: Sequence[int],
) -> tuple[int, int, int]:
    """Return ``(index, signed_to_reference_midpoint_ns, total_span_ns)``.

    The chosen timestamp minimizes the complete multi-source span.  Distance
    to the reference midpoint and recency provide deterministic tie-breaking.
    """

    if not candidate_times_ns:
        raise RuntimeError("timestamp history is empty")
    if not reference_times_ns:
        raise ValueError("at least one reference timestamp is required")
    references = [int(value) for value in reference_times_ns]
    midpoint = (min(references) + max(references)) // 2
    ranked: list[tuple[int, int, int, int]] = []
    for index, raw in enumerate(candidate_times_ns):
        value = int(raw)
        combined = [*references, value]
        span = max(combined) - min(combined)
        ranked.append((span, abs(value - midpoint), -value, index))
    span, _, _, index = min(ranked)
    value = int(candidate_times_ns[index])
    return index, value - midpoint, span


def parse_client_sync(
    raw: Any,
    *,
    server_receive_wall_time_ns: int,
    max_clock_uncertainty_ms: float,
    max_sync_skew_ms: float,
    state_sync_max_skew_ms: float = DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    expected_observation_interval_ms: float = DEFAULT_OBSERVATION_INTERVAL_MS,
    observation_interval_tolerance_ms: float = (
        DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS
    ),
) -> ParsedClientSync:
    """Validate one XR synchronization proof at the policy-server boundary.

    This deliberately revalidates client-provided state timing.  Positive
    network/transport age remains diagnostic-only; excessive clock
    uncertainty is surfaced as a typed retryable condition.
    """

    if not isinstance(raw, dict):
        raise ValueError("camera synchronization metadata is required")
    if raw.get("schema") != SYNC_SCHEMA:
        raise ValueError(f"unexpected camera sync schema: {raw}")
    if raw.get("clock_domain") != CLOCK_DOMAIN:
        raise ValueError(f"unexpected camera sync clock domain: {raw}")

    clock_sync_raw = raw.get("clock_sync")
    if (
        not isinstance(clock_sync_raw, dict)
        or clock_sync_raw.get("schema") != CLOCK_SYNC_SCHEMA
    ):
        raise ValueError("camera synchronization has no valid XR/5090 clock proof")
    clock_uncertainty_ns = clock_sync_raw.get("uncertainty_ns")
    clock_measured_server_ns = clock_sync_raw.get("measured_server_wall_time_ns")
    if (
        isinstance(clock_uncertainty_ns, bool)
        or not isinstance(clock_uncertainty_ns, int)
        or clock_uncertainty_ns < 0
    ):
        raise ValueError(f"invalid camera clock uncertainty: {clock_uncertainty_ns!r}")
    if (
        isinstance(clock_measured_server_ns, bool)
        or not isinstance(clock_measured_server_ns, int)
        or clock_measured_server_ns <= 0
    ):
        raise ValueError(
            "invalid camera clock measurement timestamp: "
            f"{clock_measured_server_ns!r}"
        )

    clock_uncertainty_ms = clock_uncertainty_ns / 1e6
    if clock_uncertainty_ms > max_clock_uncertainty_ms:
        raise ClockSyncUncertainForAlignment(
            clock_uncertainty_ms, max_clock_uncertainty_ms
        )
    clock_proof_age_ms = (
        server_receive_wall_time_ns - clock_measured_server_ns
    ) / 1e6
    if clock_proof_age_ms < -50.0:
        raise RuntimeError(
            "XR/5090 clock proof is from the future: "
            f"age={clock_proof_age_ms:.3f}ms minimum=-50.000ms"
        )

    effective_sync_limit_ms = max_sync_skew_ms - clock_uncertainty_ms
    if effective_sync_limit_ms <= 0.0:
        raise RuntimeError(
            "XR/5090 clock uncertainty consumes the complete camera sync budget: "
            f"uncertainty={clock_uncertainty_ms:.3f}ms "
            f"budget={max_sync_skew_ms:.3f}ms"
        )

    cameras_raw = raw.get("cameras")
    if not isinstance(cameras_raw, dict):
        raise ValueError("camera sync metadata has no cameras mapping")
    reference_wall_times_ns: dict[str, int] = {}
    for key in ("cam_left_wrist", "cam_right_wrist"):
        item = cameras_raw.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"camera sync metadata is missing {key}")
        timestamp = item.get("source_wall_time_ns")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp <= 0
        ):
            raise ValueError(f"invalid {key} source timestamp: {timestamp!r}")
        reference_wall_times_ns[key] = timestamp

    observation_wall_time_ns = raw.get("observation_wall_time_ns")
    if (
        isinstance(observation_wall_time_ns, bool)
        or not isinstance(observation_wall_time_ns, int)
        or observation_wall_time_ns <= 0
    ):
        raise ValueError(f"invalid observation timestamp: {observation_wall_time_ns!r}")
    expected_midpoint = (
        min(reference_wall_times_ns.values())
        + max(reference_wall_times_ns.values())
    ) // 2
    if observation_wall_time_ns != expected_midpoint:
        raise ValueError(
            "observation timestamp is not the wrist midpoint: "
            f"provided={observation_wall_time_ns} expected={expected_midpoint}"
        )

    state_first_observation_wall_time_ns = raw.get(
        "state_first_observation_wall_time_ns"
    )
    if (
        isinstance(state_first_observation_wall_time_ns, bool)
        or not isinstance(state_first_observation_wall_time_ns, int)
        or state_first_observation_wall_time_ns <= 0
    ):
        raise ValueError(
            "camera sync metadata has no valid first state observation timestamp"
        )
    state_observation_interval_ms = validate_observation_interval(
        {"observation_wall_time_ns": state_first_observation_wall_time_ns},
        {"observation_wall_time_ns": observation_wall_time_ns},
        expected_ms=expected_observation_interval_ms,
        tolerance_ms=observation_interval_tolerance_ms,
    )
    state_alignment = validate_state_alignment(
        raw.get("state_samples"),
        first_observation_wall_time_ns=state_first_observation_wall_time_ns,
        second_observation_wall_time_ns=observation_wall_time_ns,
        max_skew_ms=state_sync_max_skew_ms,
        expected_observation_interval_ms=expected_observation_interval_ms,
        observation_interval_tolerance_ms=observation_interval_tolerance_ms,
    )

    observation_transport_age_ms = (
        server_receive_wall_time_ns - observation_wall_time_ns
    ) / 1e6
    worst_observation_transport_age_ms = (
        observation_transport_age_ms + clock_uncertainty_ms
    )
    client_send_wall_time_ns = raw.get("client_send_wall_time_ns")
    source_to_send_ms = None
    send_to_receive_ms = None
    if isinstance(client_send_wall_time_ns, int) and not isinstance(
        client_send_wall_time_ns, bool
    ):
        source_to_send_ms = (
            client_send_wall_time_ns - observation_wall_time_ns
        ) / 1e6
        send_to_receive_ms = (
            server_receive_wall_time_ns - client_send_wall_time_ns
        ) / 1e6
    if observation_transport_age_ms - clock_uncertainty_ms < -50.0:
        raise RuntimeError(
            "XR/5090 wall clocks are inconsistent: "
            f"observation appears {-observation_transport_age_ms:.1f}ms in the future"
        )

    return ParsedClientSync(
        raw=raw,
        clock_sync=clock_sync_raw,
        cameras=cameras_raw,
        reference_wall_times_ns=reference_wall_times_ns,
        observation_wall_time_ns=observation_wall_time_ns,
        state_first_observation_wall_time_ns=(
            state_first_observation_wall_time_ns
        ),
        state_observation_interval_ms=state_observation_interval_ms,
        state_alignment=state_alignment,
        server_receive_wall_time_ns=server_receive_wall_time_ns,
        clock_uncertainty_ms=clock_uncertainty_ms,
        clock_proof_age_ms=clock_proof_age_ms,
        max_sync_skew_ms=max_sync_skew_ms,
        effective_sync_limit_ms=effective_sync_limit_ms,
        observation_transport_age_ms=observation_transport_age_ms,
        worst_observation_transport_age_ms=worst_observation_transport_age_ms,
        client_send_wall_time_ns=client_send_wall_time_ns,
        source_to_send_ms=source_to_send_ms,
        send_to_receive_ms=send_to_receive_ms,
    )


def build_three_camera_sync_proof(
    client: ParsedClientSync,
    *,
    e6_snapshot: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """Build the existing three-camera response proof from validated inputs."""

    all_times = {
        **client.reference_wall_times_ns,
        "cam_high": int(e6_snapshot["alignment_wall_time_ns"]),
    }
    verified_pairwise_skew_ms = (
        max(all_times.values()) - min(all_times.values())
    ) / 1e6
    if (
        verified_pairwise_skew_ms + client.clock_uncertainty_ms
        > client.max_sync_skew_ms + 1e-9
    ):
        raise RuntimeError(
            "E6 source returned a frame outside the synchronization contract: "
            f"span={verified_pairwise_skew_ms:.3f}ms "
            f"clock_uncertainty={client.clock_uncertainty_ms:.3f}ms "
            f"limit={client.max_sync_skew_ms:.3f}ms"
        )

    return {
        "schema": SYNC_SCHEMA,
        "clock_domain": CLOCK_DOMAIN,
        "status": "aligned",
        "observation_wall_time_ns": client.observation_wall_time_ns,
        "state_first_observation_wall_time_ns": (
            client.state_first_observation_wall_time_ns
        ),
        "state_observation_interval_ms": client.state_observation_interval_ms,
        "state_samples": client.raw.get("state_samples"),
        "state_alignment": client.state_alignment,
        "max_allowed_pairwise_skew_ms": client.max_sync_skew_ms,
        "effective_timestamp_skew_limit_ms": client.effective_sync_limit_ms,
        "clock_sync": dict(client.clock_sync),
        "clock_sync_age_ms": client.clock_proof_age_ms,
        "clock_uncertainty_ms": client.clock_uncertainty_ms,
        "max_pairwise_skew_ms": verified_pairwise_skew_ms,
        "signed_skew_ms": {
            key: (value - client.observation_wall_time_ns) / 1e6
            for key, value in all_times.items()
        },
        "server_receive_wall_time_ns": client.server_receive_wall_time_ns,
        "observation_transport_age_ms": client.observation_transport_age_ms,
        "worst_observation_transport_age_ms": (
            client.worst_observation_transport_age_ms
        ),
        "observation_transport_age_policy": OBSERVATION_TRANSPORT_AGE_POLICY,
        "source_to_send_ms": client.source_to_send_ms,
        "send_to_receive_ms": client.send_to_receive_ms,
        "client_send_wall_time_ns": client.client_send_wall_time_ns,
        "timestamp_quality": {
            "cam_high": e6_snapshot["timestamp_semantics"],
            "cam_left_wrist": client.cameras["cam_left_wrist"].get(
                "timestamp_semantics"
            ),
            "cam_right_wrist": client.cameras["cam_right_wrist"].get(
                "timestamp_semantics"
            ),
        },
        "cameras": {
            **{
                key: dict(client.cameras[key])
                for key in client.reference_wall_times_ns
            },
            "cam_high": dict(e6_snapshot),
        },
        "selection": alignment,
    }
