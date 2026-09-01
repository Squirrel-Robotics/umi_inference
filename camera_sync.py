#!/usr/bin/env python3
"""Pure helpers for fail-closed multi-camera timestamp alignment.

All cross-host timestamps in this module are Unix epoch nanoseconds.  A
``monotonic_ns`` value is only used to decide whether a local stream is alive;
it is never compared between XR and the policy host.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


SYNC_SCHEMA = "umi_camera_sync_v3"
XR_CLOCK_DOMAIN = "xr_unix_epoch_ns"
CLOCK_DOMAIN = "policy_server_unix_epoch_ns_estimated_v1"
CLOCK_SYNC_SCHEMA = "umi_min_rtt_clock_sync_v1"
OBSERVATION_RETRY_SCHEMA = "umi_observation_retry_v1"
OBSERVATION_TRANSPORT_AGE_POLICY = "diagnostic_only"
DEFAULT_OBSERVATION_INTERVAL_MS = 100.0
DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS = 45.0
DEFAULT_STATE_SYNC_MAX_SKEW_MS = 25.0


@dataclass(frozen=True)
class TimedCameraFrame:
    value: Any
    source_wall_time_ns: int
    arrival_wall_time_ns: int
    arrival_monotonic_ns: int
    sequence: int
    source: str
    timestamp_semantics: str
    transport_value: Any | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "source_wall_time_ns": int(self.source_wall_time_ns),
            "arrival_wall_time_ns": int(self.arrival_wall_time_ns),
            "arrival_monotonic_ns": int(self.arrival_monotonic_ns),
            "sequence": int(self.sequence),
            "source": self.source,
            "timestamp_semantics": self.timestamp_semantics,
        }


def _fresh_frames(
    frames: Iterable[TimedCameraFrame],
    *,
    now_monotonic_ns: int,
    max_age_ns: int,
) -> list[TimedCameraFrame]:
    return [
        frame
        for frame in frames
        if 0 <= now_monotonic_ns - frame.arrival_monotonic_ns <= max_age_ns
    ]


def select_latest_aligned_pair(
    left_frames: Iterable[TimedCameraFrame],
    right_frames: Iterable[TimedCameraFrame],
    *,
    now_monotonic_ns: int,
    max_age_ns: int,
    max_skew_ns: int,
    after_observation_wall_time_ns: int | None = None,
    after_sequences: dict[str, int] | None = None,
) -> tuple[TimedCameraFrame, TimedCameraFrame, dict[str, Any]]:
    """Return the newest fresh wrist pair whose source stamps are aligned.

    Selection is based on source/capture time, never recorder file time.  The
    newest valid midpoint wins; a smaller skew breaks ties.  There is no
    fallback to an unaligned latest frame.
    """

    if max_age_ns <= 0 or max_skew_ns < 0:
        raise ValueError("camera age/skew limits must be positive")
    left = _fresh_frames(
        left_frames, now_monotonic_ns=now_monotonic_ns, max_age_ns=max_age_ns
    )
    right = _fresh_frames(
        right_frames, now_monotonic_ns=now_monotonic_ns, max_age_ns=max_age_ns
    )
    if not left or not right:
        missing = []
        if not left:
            missing.append("left")
        if not right:
            missing.append("right")
        raise RuntimeError("no fresh wrist frame: " + ",".join(missing))

    candidates: list[
        tuple[int, int, TimedCameraFrame, TimedCameraFrame]
    ] = []
    for left_frame in left:
        for right_frame in right:
            if after_sequences is not None and (
                left_frame.sequence
                <= int(after_sequences.get("cam_left_wrist", -1))
                or right_frame.sequence
                <= int(after_sequences.get("cam_right_wrist", -1))
            ):
                continue
            signed_skew_ns = (
                right_frame.source_wall_time_ns - left_frame.source_wall_time_ns
            )
            if abs(signed_skew_ns) > max_skew_ns:
                continue
            observation_ns = (
                left_frame.source_wall_time_ns + right_frame.source_wall_time_ns
            ) // 2
            if (
                after_observation_wall_time_ns is not None
                and observation_ns <= after_observation_wall_time_ns
            ):
                continue
            candidates.append(
                (observation_ns, -abs(signed_skew_ns), left_frame, right_frame)
            )
    if not candidates:
        nearest_ns = min(
            abs(r.source_wall_time_ns - l.source_wall_time_ns)
            for l in left
            for r in right
        )
        qualifier = (
            " and using new frames after the previous observation"
            if after_observation_wall_time_ns is not None or after_sequences is not None
            else ""
        )
        raise RuntimeError(
            "no aligned wrist pair"
            f"{qualifier}: nearest_skew_ms={nearest_ns / 1e6:.3f} "
            f"limit_ms={max_skew_ns / 1e6:.3f}"
        )

    observation_ns, _, left_frame, right_frame = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    signed_skew_ns = right_frame.source_wall_time_ns - left_frame.source_wall_time_ns
    metadata = {
        "schema": SYNC_SCHEMA,
        "clock_domain": XR_CLOCK_DOMAIN,
        "status": "wrist_aligned",
        "observation_wall_time_ns": int(observation_ns),
        "max_allowed_pairwise_skew_ms": max_skew_ns / 1e6,
        "wrist_signed_skew_ms": signed_skew_ns / 1e6,
        "wrist_pairwise_skew_ms": abs(signed_skew_ns) / 1e6,
        "cameras": {
            "cam_left_wrist": left_frame.metadata(),
            "cam_right_wrist": right_frame.metadata(),
        },
    }
    return left_frame, right_frame, metadata


def translate_sync_to_policy_clock(
    sync: dict[str, Any],
    clock_sync: dict[str, Any],
) -> dict[str, Any]:
    """Translate an XR timestamp proof into the policy host's wall clock.

    Local selection must remain in the XR clock domain.  This function returns
    a deep copy and shifts only positive integer fields whose names end in
    ``_wall_time_ns``; monotonic timestamps, durations, sequences, and the
    original object are never changed.
    """

    if not isinstance(sync, dict) or sync.get("schema") != SYNC_SCHEMA:
        raise ValueError(f"unexpected local camera sync schema: {sync!r}")
    if sync.get("clock_domain") != XR_CLOCK_DOMAIN:
        raise ValueError(
            "camera sync is not in the XR clock domain or was already translated: "
            f"{sync.get('clock_domain')!r}"
        )
    if not isinstance(clock_sync, dict) or clock_sync.get("schema") != CLOCK_SYNC_SCHEMA:
        raise ValueError(f"invalid XR/5090 clock proof: {clock_sync!r}")
    offset_ns = clock_sync.get("server_minus_client_wall_time_ns")
    uncertainty_ns = clock_sync.get("uncertainty_ns")
    measured_client_ns = clock_sync.get("measured_client_wall_time_ns")
    for label, value in (
        ("server_minus_client_wall_time_ns", offset_ns),
        ("uncertainty_ns", uncertainty_ns),
        ("measured_client_wall_time_ns", measured_client_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid clock proof {label}: {value!r}")
    if uncertainty_ns < 0 or measured_client_ns <= 0:
        raise ValueError(f"invalid clock proof values: {clock_sync!r}")

    shifted_fields = 0

    def translate(value: Any, key: str | None = None) -> Any:
        nonlocal shifted_fields
        if isinstance(value, dict):
            return {
                str(child_key): translate(child_value, str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [translate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(translate(item) for item in value)
        if key is not None and key.endswith("_wall_time_ns"):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"wall timestamp {key} must be an integer: {value!r}")
            if value <= 0:
                raise ValueError(f"wall timestamp {key} must be positive: {value!r}")
            shifted = value + offset_ns
            if shifted <= 0:
                raise ValueError(
                    f"translated wall timestamp {key} is not positive: {shifted}"
                )
            shifted_fields += 1
            return shifted
        return value

    result = translate(sync)
    result["clock_domain"] = CLOCK_DOMAIN
    result["source_clock_domain"] = XR_CLOCK_DOMAIN
    result["clock_sync"] = {
        **clock_sync,
        "measured_server_wall_time_ns": measured_client_ns + offset_ns,
        "translated_wall_time_field_count": shifted_fields,
    }
    return result


def select_nearest_timed_sample(
    frames: Iterable[TimedCameraFrame],
    *,
    target_wall_time_ns: int,
    now_monotonic_ns: int,
    max_age_ns: int,
    max_skew_ns: int,
) -> tuple[TimedCameraFrame, dict[str, Any]]:
    """Select one fresh state sample nearest to an image observation anchor."""

    fresh = _fresh_frames(
        frames, now_monotonic_ns=now_monotonic_ns, max_age_ns=max_age_ns
    )
    if not fresh:
        raise RuntimeError("no fresh timestamped sample")
    selected = min(
        fresh,
        key=lambda frame: (
            abs(frame.source_wall_time_ns - target_wall_time_ns),
            -frame.source_wall_time_ns,
        ),
    )
    signed_skew_ns = selected.source_wall_time_ns - int(target_wall_time_ns)
    if abs(signed_skew_ns) > max_skew_ns:
        raise RuntimeError(
            "no state sample aligned to the image observation: "
            f"nearest_skew_ms={abs(signed_skew_ns) / 1e6:.3f} "
            f"limit_ms={max_skew_ns / 1e6:.3f}"
        )
    metadata = selected.metadata()
    metadata["signed_to_observation_ms"] = signed_skew_ns / 1e6
    return selected, metadata


def validate_observation_interval(
    first_sync: dict[str, Any],
    second_sync: dict[str, Any],
    *,
    expected_ms: float = DEFAULT_OBSERVATION_INTERVAL_MS,
    tolerance_ms: float = DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS,
) -> float:
    """Validate the two camera-anchored observations used to build state30.

    Merely sleeping for one nominal period doesn't prove that the selected
    source frames have the training-time cadence. This check uses retained
    source-time midpoints and fails closed on buffered or delayed pairs.
    """

    if expected_ms <= 0.0 or tolerance_ms < 0.0 or tolerance_ms >= expected_ms:
        raise ValueError("invalid observation interval contract")
    first_ns = first_sync.get("observation_wall_time_ns")
    second_ns = second_sync.get("observation_wall_time_ns")
    for label, value in (("first", first_ns), ("second", second_ns)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid {label} observation timestamp: {value!r}")
    interval_ms = (second_ns - first_ns) / 1e6
    lower = expected_ms - tolerance_ms
    upper = expected_ms + tolerance_ms
    if not lower <= interval_ms <= upper:
        raise RuntimeError(
            "state observation interval is outside the policy cadence: "
            f"actual={interval_ms:.3f}ms allowed=[{lower:.3f},{upper:.3f}]ms"
        )
    return interval_ms


def validate_eef_sample_intervals(
    first_timing: dict[str, Any],
    second_timing: dict[str, Any],
    *,
    expected_ms: float = DEFAULT_OBSERVATION_INTERVAL_MS,
    tolerance_ms: float = DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS,
) -> dict[str, float]:
    """Require both EEF samples to advance at the reviewed policy cadence."""

    if expected_ms <= 0.0 or tolerance_ms < 0.0 or tolerance_ms >= expected_ms:
        raise ValueError("invalid EEF interval contract")
    lower = expected_ms - tolerance_ms
    upper = expected_ms + tolerance_ms
    intervals: dict[str, float] = {}
    for key in ("left_eef", "right_eef"):
        first = first_timing.get(key)
        second = second_timing.get(key)
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError(f"missing {key} timing metadata")
        first_sequence = first.get("sequence")
        second_sequence = second.get("sequence")
        first_ns = first.get("source_wall_time_ns")
        second_ns = second.get("source_wall_time_ns")
        values = (first_sequence, second_sequence, first_ns, second_ns)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"invalid {key} timing metadata: {first}, {second}")
        if second_sequence <= first_sequence:
            raise RuntimeError(f"{key} did not advance between state observations")
        interval_ms = (second_ns - first_ns) / 1e6
        if not lower <= interval_ms <= upper:
            raise RuntimeError(
                f"{key} interval is outside the policy cadence: "
                f"actual={interval_ms:.3f}ms allowed=[{lower:.3f},{upper:.3f}]ms"
            )
        intervals[key] = interval_ms
    return intervals


def validate_state_alignment(
    state_samples: dict[str, Any],
    *,
    first_observation_wall_time_ns: int,
    second_observation_wall_time_ns: int,
    max_skew_ms: float = DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    expected_observation_interval_ms: float = DEFAULT_OBSERVATION_INTERVAL_MS,
    observation_interval_tolerance_ms: float = (
        DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS
    ),
) -> dict[str, Any]:
    """Server-verifiable proof that robot state samples match image anchors."""

    if not isinstance(state_samples, dict):
        raise ValueError("camera sync metadata has no state_samples mapping")
    first = state_samples.get("first")
    second = state_samples.get("second")
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("state_samples must contain first and second mappings")
    if max_skew_ms <= 0.0:
        raise ValueError("state alignment limit must be positive")
    limit_ns = int(round(max_skew_ms * 1e6))
    signed_skews_ms: dict[str, dict[str, float]] = {"first": {}, "second": {}}
    for phase, samples, target_ns in (
        ("first", first, first_observation_wall_time_ns),
        ("second", second, second_observation_wall_time_ns),
    ):
        for key in ("left_eef", "right_eef", "left_hand", "right_hand"):
            sample = samples.get(key)
            if not isinstance(sample, dict):
                raise ValueError(f"state_samples.{phase} is missing {key}")
            source_ns = sample.get("source_wall_time_ns")
            if isinstance(source_ns, bool) or not isinstance(source_ns, int):
                raise ValueError(
                    f"invalid state_samples.{phase}.{key} timestamp: {source_ns!r}"
                )
            signed_ns = source_ns - int(target_ns)
            if abs(signed_ns) > limit_ns:
                raise RuntimeError(
                    f"{phase} {key} is not aligned to the image observation: "
                    f"skew={signed_ns / 1e6:.3f}ms limit={max_skew_ms:.3f}ms"
                )
            signed_skews_ms[phase][key] = signed_ns / 1e6
    eef_intervals_ms = validate_eef_sample_intervals(
        first,
        second,
        expected_ms=expected_observation_interval_ms,
        tolerance_ms=observation_interval_tolerance_ms,
    )
    return {
        "max_allowed_state_skew_ms": float(max_skew_ms),
        "signed_skew_ms": signed_skews_ms,
        "eef_intervals_ms": eef_intervals_ms,
    }


def validate_camera_sync_response(
    raw: Any,
    *,
    expected_observation_wall_time_ns: int,
    clock_uncertainty_ms: float,
    max_camera_skew_ms: float,
    max_state_skew_ms: float = DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    expected_observation_interval_ms: float = DEFAULT_OBSERVATION_INTERVAL_MS,
    observation_interval_tolerance_ms: float = (
        DEFAULT_OBSERVATION_INTERVAL_TOLERANCE_MS
    ),
) -> dict[str, Any]:
    """Fail closed on a server's alignment receipt before robot execution."""

    if not isinstance(raw, dict) or raw.get("status") != "aligned":
        raise RuntimeError(f"policy response has no aligned camera proof: {raw}")
    if raw.get("schema") != SYNC_SCHEMA or raw.get("clock_domain") != CLOCK_DOMAIN:
        raise RuntimeError(
            "policy response camera-proof schema/clock domain mismatch: "
            f"schema={raw.get('schema')!r} clock_domain={raw.get('clock_domain')!r}"
        )
    if raw.get("observation_wall_time_ns") != expected_observation_wall_time_ns:
        raise RuntimeError(
            "policy camera synchronization correlation mismatch: "
            f"expected={expected_observation_wall_time_ns} actual={raw}"
        )

    response_observation_interval_ms = validate_observation_interval(
        {
            "observation_wall_time_ns": raw.get(
                "state_first_observation_wall_time_ns"
            )
        },
        {"observation_wall_time_ns": expected_observation_wall_time_ns},
        expected_ms=expected_observation_interval_ms,
        tolerance_ms=observation_interval_tolerance_ms,
    )

    def finite(name: str, value: Any, *, minimum: float | None = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid {name}: {value!r}") from exc
        if not math.isfinite(number) or (minimum is not None and number < minimum):
            raise RuntimeError(f"invalid {name}: {value!r}")
        return number

    camera_skew_ms = finite(
        "camera max_pairwise_skew_ms",
        raw.get("max_pairwise_skew_ms"),
        minimum=0.0,
    )
    clock_ms = finite("clock uncertainty", clock_uncertainty_ms, minimum=0.0)
    camera_limit_ms = finite(
        "camera sync limit", max_camera_skew_ms, minimum=0.0
    )
    if camera_limit_ms <= 0.0 or camera_skew_ms + clock_ms > camera_limit_ms + 1e-9:
        raise RuntimeError(
            "policy camera skew plus clock uncertainty exceeds XR limit: "
            f"span={camera_skew_ms}ms uncertainty={clock_ms}ms "
            f"limit={camera_limit_ms}ms"
        )

    transport_age_ms = finite(
        "observation transport age",
        raw.get("observation_transport_age_ms"),
    )
    reported_observation_interval_ms = finite(
        "reported state observation interval",
        raw.get("state_observation_interval_ms"),
    )
    if not math.isclose(
        reported_observation_interval_ms,
        response_observation_interval_ms,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "policy response state observation interval does not match its "
            "timestamps: "
            f"reported={reported_observation_interval_ms}ms "
            f"actual={response_observation_interval_ms}ms"
        )
    if raw.get("observation_transport_age_policy") != OBSERVATION_TRANSPORT_AGE_POLICY:
        raise RuntimeError(
            "policy response changed transport-age behavior: "
            f"{raw.get('observation_transport_age_policy')!r}"
        )

    state = raw.get("state_alignment")
    if not isinstance(state, dict):
        raise RuntimeError("policy response has no state-alignment proof")
    state_limit_ms = finite(
        "state alignment limit",
        state.get("max_allowed_state_skew_ms"),
        minimum=0.0,
    )
    required_state_limit_ms = finite(
        "required state alignment limit", max_state_skew_ms, minimum=0.0
    )
    if (
        state_limit_ms <= 0.0
        or required_state_limit_ms <= 0.0
        or state_limit_ms > required_state_limit_ms + 1e-9
    ):
        raise RuntimeError(
            "policy state alignment limit is weaker than XR: "
            f"server={state_limit_ms}ms xr={required_state_limit_ms}ms"
        )
    signed = state.get("signed_skew_ms")
    if not isinstance(signed, dict):
        raise RuntimeError("policy response has no state signed-skew proof")
    for phase in ("first", "second"):
        samples = signed.get(phase)
        if not isinstance(samples, dict):
            raise RuntimeError(f"policy state proof is missing {phase}")
        for key in ("left_eef", "right_eef", "left_hand", "right_hand"):
            skew_ms = finite(f"{phase} {key} state skew", samples.get(key))
            if abs(skew_ms) > state_limit_ms + 1e-9:
                raise RuntimeError(
                    f"policy {phase} {key} state skew exceeds its proof: "
                    f"skew={skew_ms}ms limit={state_limit_ms}ms"
                )
    intervals = state.get("eef_intervals_ms")
    if not isinstance(intervals, dict):
        raise RuntimeError("policy response has no EEF interval proof")
    if (
        expected_observation_interval_ms <= 0.0
        or observation_interval_tolerance_ms < 0.0
        or observation_interval_tolerance_ms >= expected_observation_interval_ms
    ):
        raise ValueError("invalid observation interval contract")
    lower = expected_observation_interval_ms - observation_interval_tolerance_ms
    upper = expected_observation_interval_ms + observation_interval_tolerance_ms
    for key in ("left_eef", "right_eef"):
        interval_ms = finite(f"{key} interval", intervals.get(key))
        if not lower <= interval_ms <= upper:
            raise RuntimeError(
                f"policy {key} interval is outside the observation contract: "
                f"actual={interval_ms}ms allowed=[{lower},{upper}]ms"
            )

    # Keep transport age available to the caller for logging without making
    # positive network delay an execution cutoff.
    del transport_age_ms
    return raw
