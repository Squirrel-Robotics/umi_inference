#!/usr/bin/env python3
"""Time-preserving interpolation for absolute XR arm pose commands."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ArmInterpolationPoint:
    substep: int
    substeps: int
    offset_seconds: float
    alpha: float
    left_pose7: np.ndarray
    right_pose7: np.ndarray


class ChunkLeftZExecutionAnchor:
    """Apply one base-link Z execution-anchor shift to every chunk target.

    ``prepare`` returns a copy and never mutates the decoded model target.  The
    returned boolean identifies the first target actually dispatched for a new
    chunk.  The caller commits that sequence only after the final arm endpoint
    has been accepted, so a stop or SDK failure cannot consume the transition.

    Every later target in the committed chunk receives the same translation.
    This is equivalent to shifting that chunk's execution anchor and preserves
    every consecutive Delta EEF translation exactly, without a +8 mm recovery
    on the second action.
    """

    def __init__(self, offset_m: float = -0.008) -> None:
        if isinstance(offset_m, bool) or not isinstance(offset_m, (int, float)):
            raise ValueError(f"offset_m must be finite: {offset_m!r}")
        self.offset_m = float(offset_m)
        if not math.isfinite(self.offset_m):
            raise ValueError(f"offset_m must be finite: {offset_m!r}")
        self._committed_sequence: int | None = None

    @property
    def committed_sequence(self) -> int | None:
        return self._committed_sequence

    @staticmethod
    def _sequence(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"chunk_sequence must be an integer: {value!r}")
        sequence = int(value)
        if sequence < 0:
            raise ValueError(
                f"chunk_sequence must be nonnegative: {value!r}"
            )
        return sequence

    def prepare(
        self, chunk_sequence: int, decoded_left_pose7: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Return the shifted target and whether it starts a new chunk."""

        sequence = self._sequence(chunk_sequence)
        if (
            self._committed_sequence is not None
            and sequence < self._committed_sequence
        ):
            raise ValueError(
                "chunk sequence moved backwards: "
                f"committed={self._committed_sequence} current={sequence}"
            )
        target = _validated_pose7("decoded_left_pose7", decoded_left_pose7)
        first_action = sequence != self._committed_sequence
        # The decoded pose is absolute in base_link. Applying the same base-Z
        # shift to every target is equivalent to left-multiplying the chunk's
        # shared execution anchor by Trans_base(0, 0, offset_m). The model
        # Delta EEF rows and all target-to-target differences remain unchanged.
        target[2] += self.offset_m
        return target, first_action

    def commit(self, chunk_sequence: int) -> None:
        """Mark this chunk's first action complete after endpoint acceptance."""

        sequence = self._sequence(chunk_sequence)
        if (
            self._committed_sequence is not None
            and sequence < self._committed_sequence
        ):
            raise ValueError(
                "chunk sequence moved backwards: "
                f"committed={self._committed_sequence} current={sequence}"
            )
        self._committed_sequence = sequence


def next_dispatch_deadline(
    *,
    now: float,
    due: float,
    next_due: float | None,
    last_dispatch_started: float | None,
    arm_command_rate_hz: float,
) -> float | None:
    """Choose one non-bursting SDK dispatch deadline.

    A late intermediate point is discarded when it can no longer fit one full
    arm-command period before the following point.  The final endpoint has no
    successor and is therefore always retained.  This prevents catch-up bursts
    without changing which model endpoint closes the action segment.
    """

    command_period = 1.0 / _positive_rate(
        "arm_command_rate_hz", arm_command_rate_hz
    )
    values = (now, due)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("dispatch times must be finite")
    if next_due is not None and not math.isfinite(float(next_due)):
        raise ValueError("next_due must be finite when supplied")
    if last_dispatch_started is not None and not math.isfinite(
        float(last_dispatch_started)
    ):
        raise ValueError("last_dispatch_started must be finite when supplied")
    earliest = max(float(now), float(due))
    if last_dispatch_started is not None:
        earliest = max(
            earliest, float(last_dispatch_started) + command_period
        )
    # Permit normal scheduler jitter without collapsing a nominal 50 Hz
    # stream to every other point.  This slack never permits a burst because
    # ``earliest`` still enforces one complete command period from the last
    # dispatch start; it only lets the remaining schedule shift slightly.
    tolerance = max(1e-12, min(0.002, command_period * 0.1))
    if (
        next_due is not None
        and earliest + command_period > float(next_due) + tolerance
    ):
        return None
    return earliest


def _positive_rate(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number: {value!r}")
    return number


def interpolation_offsets(
    *,
    action_rate_hz: float,
    arm_command_rate_hz: float,
) -> tuple[float, ...]:
    """Return command offsets that end exactly at one action waypoint.

    Intermediate commands follow the requested arm period.  The original
    action endpoint is always appended exactly once, including when the two
    rates are not integer multiples or the command rate is lower.
    """

    action_rate = _positive_rate("action_rate_hz", action_rate_hz)
    command_rate = _positive_rate(
        "arm_command_rate_hz", arm_command_rate_hz
    )
    duration = 1.0 / action_rate
    command_period = 1.0 / command_rate
    tolerance = max(1e-12, duration * 1e-12)
    offsets: list[float] = []
    substep = 1
    while True:
        offset = substep * command_period
        if offset >= duration - tolerance:
            break
        offsets.append(offset)
        substep += 1
    offsets.append(duration)
    return tuple(offsets)


def _validated_pose7(name: str, pose7: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite pose7")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if not math.isfinite(quaternion_norm) or quaternion_norm <= 1e-12:
        raise ValueError(f"{name} has an invalid quaternion")
    if abs(quaternion_norm - 1.0) > 1e-5:
        raise ValueError(
            f"{name} quaternion must be normalized: norm={quaternion_norm}"
        )
    return pose.copy()


def quaternion_slerp_xyzw(
    start_xyzw: np.ndarray,
    target_xyzw: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shortest-arc quaternion SLERP with a stable near-linear branch."""

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be within [0, 1]: {alpha!r}")
    fraction = float(alpha)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"alpha must be within [0, 1]: {alpha!r}")
    start = np.asarray(start_xyzw, dtype=np.float64)
    target = np.asarray(target_xyzw, dtype=np.float64)
    if start.shape != (4,) or target.shape != (4,):
        raise ValueError("quaternion inputs must have shape (4,)")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("quaternion inputs must be finite")
    start_norm = float(np.linalg.norm(start))
    target_norm = float(np.linalg.norm(target))
    if start_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("quaternion inputs must be nonzero")
    start = start / start_norm
    target = target / target_norm
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = start + fraction * (target - start)
        result /= np.linalg.norm(result)
        return result
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    start_weight = math.sin((1.0 - fraction) * theta) / sin_theta
    target_weight = math.sin(fraction * theta) / sin_theta
    result = start_weight * start + target_weight * target
    result /= np.linalg.norm(result)
    return result


def _quaternion_multiply_xyzw(
    left_xyzw: np.ndarray, right_xyzw: np.ndarray
) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left_xyzw, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right_xyzw, dtype=np.float64)
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quaternion_matrix_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion must be nonzero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def interpolate_pose7(
    start_pose7: np.ndarray,
    target_pose7: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate a right-relative pose increment between absolute targets.

    The model delta has already been decoded exactly once as
    ``T_target = T_anchor @ Delta``.  For smoothing, this function computes
    ``D = inv(T_start) @ T_target`` and returns
    ``T_start @ Interp(I, D, alpha)``.  Translation and rotation are
    interpolated separately in the start/body frame used by the action
    representation; no model delta is accumulated again.
    """

    start = _validated_pose7("start_pose7", start_pose7)
    target = _validated_pose7("target_pose7", target_pose7)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be within [0, 1]: {alpha!r}")
    fraction = float(alpha)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"alpha must be within [0, 1]: {alpha!r}")
    if fraction == 0.0:
        return start
    # q and -q encode the same rotation. Keep the target in the hemisphere of
    # the last command so the numeric SDK stream cannot jump signs at alpha=1.
    if float(np.dot(start[3:], target[3:])) < 0.0:
        target[3:] *= -1.0
    if fraction == 1.0:
        # Translation is exact and rotation is physically identical to the
        # decoded endpoint; only the quaternion's redundant sign may differ.
        return target
    result = np.empty(7, dtype=np.float64)
    start_rotation = _quaternion_matrix_xyzw(start[3:])
    relative_translation = start_rotation.T @ (target[:3] - start[:3])
    result[:3] = start[:3] + start_rotation @ (
        fraction * relative_translation
    )

    start_quaternion = start[3:] / np.linalg.norm(start[3:])
    target_quaternion = target[3:] / np.linalg.norm(target[3:])
    start_inverse = np.array(
        [-start_quaternion[0], -start_quaternion[1], -start_quaternion[2], start_quaternion[3]],
        dtype=np.float64,
    )
    relative_quaternion = _quaternion_multiply_xyzw(
        start_inverse, target_quaternion
    )
    relative_step = quaternion_slerp_xyzw(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        relative_quaternion,
        fraction,
    )
    result[3:] = _quaternion_multiply_xyzw(
        start_quaternion, relative_step
    )
    result[3:] /= np.linalg.norm(result[3:])
    return result


def interpolate_arm_segment(
    *,
    start_left_pose7: np.ndarray,
    start_right_pose7: np.ndarray,
    target_left_pose7: np.ndarray,
    target_right_pose7: np.ndarray,
    action_rate_hz: float,
    arm_command_rate_hz: float,
) -> tuple[ArmInterpolationPoint, ...]:
    """Build one time-preserving segment ending at the model waypoint."""

    offsets = interpolation_offsets(
        action_rate_hz=action_rate_hz,
        arm_command_rate_hz=arm_command_rate_hz,
    )
    duration = 1.0 / _positive_rate("action_rate_hz", action_rate_hz)
    points: list[ArmInterpolationPoint] = []
    for index, offset in enumerate(offsets, start=1):
        alpha = 1.0 if index == len(offsets) else offset / duration
        points.append(
            ArmInterpolationPoint(
                substep=index,
                substeps=len(offsets),
                offset_seconds=offset,
                alpha=alpha,
                left_pose7=interpolate_pose7(
                    start_left_pose7, target_left_pose7, alpha
                ),
                right_pose7=interpolate_pose7(
                    start_right_pose7, target_right_pose7, alpha
                ),
            )
        )
    return tuple(points)
