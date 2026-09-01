#!/usr/bin/env python3
"""Pure timing helpers for asynchronous UMI action-chunk scheduling.

This module deliberately has no ROS, robot SDK, WebSocket, NumPy, or wall
clock dependencies.  The caller supplies XR-local monotonic timestamps.  Raw
policy action ``k`` is treated as the target at ``(k + 1) / policy_rate``
seconds after the observation anchor.

``chunk_size`` is a prefix length in *execution positions*.  Therefore a
time-aligned chunk beginning at position 3 with ``chunk_size=40`` has 37
usable prefix targets left; it does not extend its boundary to position 43.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Generic, Protocol, Sequence, TypeVar


NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_MAX_FUTURE_WALL_TIME_NS = 50_000_000


class SchedulableChunk(Protocol):
    """Minimal policy-chunk timing contract used by the buffer."""

    sequence: int
    execution_indices: Sequence[int]
    observation_monotonic_ns: int
    policy_rate_hz: float
    schedule_explicit: bool


ChunkT = TypeVar("ChunkT", bound=SchedulableChunk)


@dataclass(frozen=True)
class ExecutionSelection:
    """The first scheduled policy target that has not expired."""

    execution_position: int
    raw_policy_index: int
    target_monotonic_ns: int


@dataclass(frozen=True)
class ChunkPrefixPlan:
    """One time-aligned executable prefix and its prefetch boundary."""

    start_position: int
    prefetch_position: int
    planned_boundary_position: int
    usable_tail_positions: tuple[int, ...]
    usable_tail_raw_indices: tuple[int, ...]
    steps_until_prefetch: int
    usable_tail_steps: int
    tail_steps_at_prefetch: int

    @property
    def prefetch_due(self) -> bool:
        return self.start_position >= self.prefetch_position

    @property
    def prefix_exhausted(self) -> bool:
        return self.usable_tail_steps == 0


@dataclass(frozen=True)
class ChunkSwitchDecision:
    """Result of one non-blocking standby-to-active switch attempt."""

    status: str
    candidate_sequence: int | None = None
    old_sequence: int | None = None
    old_cursor: int | None = None
    old_tail_steps_used: int = 0
    selection: ExecutionSelection | None = None
    observation_age_ns: int | None = None
    standby_wait_ns: int | None = None
    timing_rate_hz: float | None = None

    @property
    def swapped(self) -> bool:
        return self.status == "swapped"

    @property
    def stale(self) -> bool:
        return self.status == "stale"


@dataclass(frozen=True)
class PeriodicDeadlineUpdate:
    next_deadline: float
    overrun: bool
    lateness_seconds: float


@dataclass
class AsyncChunkBuffer(Generic[ChunkT]):
    """Pure single-pending/double-buffer chunk state machine.

    The executor owns the actual Future and calls :meth:`mark_request_launched`
    exactly when it submits that Future.  This object owns every other chunk
    transition so the active chunk, its cursor, and the standby chunk cannot be
    updated independently.
    """

    active: ChunkT
    chunk_size: int
    lead_steps: int
    executable_horizon: int
    cursor: int = 0
    request_launched_for_sequence: int | None = None
    standby: ChunkT | None = None
    standby_ready_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        schedule = validated_execution_indices(self.active.execution_indices)
        horizon = _require_plain_int(
            "executable_horizon", self.executable_horizon
        )
        if horizon != len(schedule):
            raise ValueError(
                "executable_horizon must equal the execution schedule length; "
                f"got horizon={horizon}, length={len(schedule)}"
            )
        plan_chunk_prefix(
            execution_indices=schedule,
            chunk_size=self.chunk_size,
            lead_steps=self.lead_steps,
            start_position=self.cursor,
        )
        self._validate_chunk_timing(self.active, "active")

    @property
    def plan(self) -> ChunkPrefixPlan:
        return plan_chunk_prefix(
            execution_indices=self.active.execution_indices,
            chunk_size=self.chunk_size,
            lead_steps=self.lead_steps,
            start_position=self.cursor,
        )

    @property
    def at_or_after_boundary(self) -> bool:
        return self.cursor >= self.chunk_size

    @property
    def in_old_tail(self) -> bool:
        return self.chunk_size <= self.cursor < self.executable_horizon

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.executable_horizon

    def should_launch(self, *, pending_exists: bool) -> bool:
        return (
            not pending_exists
            and self.standby is None
            and self.request_launched_for_sequence != self.active.sequence
            and self.cursor >= self.plan.prefetch_position
        )

    def mark_request_launched(self) -> None:
        if self.standby is not None:
            raise RuntimeError("cannot launch while a standby chunk is buffered")
        if self.request_launched_for_sequence is not None:
            raise RuntimeError(
                "replacement request was already launched for active sequence "
                f"{self.request_launched_for_sequence}"
            )
        self.request_launched_for_sequence = self.active.sequence

    def accept_ready(
        self, candidate: ChunkT, *, ready_monotonic_ns: int
    ) -> None:
        if self.standby is not None:
            raise RuntimeError("standby buffer is already occupied")
        if self.request_launched_for_sequence != self.active.sequence:
            raise RuntimeError(
                "ready chunk has no matching active-sequence request: "
                f"active={self.active.sequence} "
                f"launched_for={self.request_launched_for_sequence}"
            )
        ready_ns = _require_plain_int(
            "ready_monotonic_ns", ready_monotonic_ns
        )
        if ready_ns <= 0:
            raise ValueError("ready_monotonic_ns must be positive")
        self._validate_chunk_timing(candidate, "standby")
        if candidate.sequence <= self.active.sequence:
            raise ValueError(
                "standby sequence must be newer than active sequence: "
                f"active={self.active.sequence} standby={candidate.sequence}"
            )
        active_schedule = tuple(self.active.execution_indices)
        candidate_schedule = tuple(candidate.execution_indices)
        if candidate_schedule != active_schedule:
            raise ValueError(
                "standby execution schedule changed: "
                f"expected={active_schedule} actual={candidate_schedule}"
            )
        self.standby = candidate
        self.standby_ready_monotonic_ns = ready_ns

    def try_switch(
        self,
        *,
        next_send_deadline_ns: int,
        control_rate_hz: float,
    ) -> ChunkSwitchDecision:
        """Switch a ready chunk at/after the boundary without blocking."""

        if not self.at_or_after_boundary:
            return ChunkSwitchDecision(status="before_boundary")
        if self.standby is None:
            return ChunkSwitchDecision(status="waiting")

        deadline_ns = _require_plain_int(
            "next_send_deadline_ns", next_send_deadline_ns
        )
        candidate = self.standby
        timing_rate_hz = effective_action_timing_rate_hz(
            control_rate_hz=control_rate_hz,
            policy_rate_hz=candidate.policy_rate_hz,
            schedule_explicit=candidate.schedule_explicit,
        )
        selection = select_first_unexpired_action(
            policy_rate_hz=timing_rate_hz,
            execution_indices=candidate.execution_indices,
            observation_monotonic_ns=candidate.observation_monotonic_ns,
            next_send_deadline_ns=deadline_ns,
        )
        candidate_sequence = candidate.sequence
        observation_age_ns = deadline_ns - candidate.observation_monotonic_ns
        standby_wait_ns = (
            None
            if self.standby_ready_monotonic_ns is None
            else deadline_ns - self.standby_ready_monotonic_ns
        )
        if selection is None or selection.execution_position >= self.chunk_size:
            self.standby = None
            self.standby_ready_monotonic_ns = None
            self.request_launched_for_sequence = None
            return ChunkSwitchDecision(
                status="stale",
                candidate_sequence=candidate_sequence,
                selection=selection,
                observation_age_ns=observation_age_ns,
                standby_wait_ns=standby_wait_ns,
                timing_rate_hz=timing_rate_hz,
            )

        old_sequence = self.active.sequence
        old_cursor = self.cursor
        self.active = candidate
        self.cursor = selection.execution_position
        self.standby = None
        self.standby_ready_monotonic_ns = None
        self.request_launched_for_sequence = None
        return ChunkSwitchDecision(
            status="swapped",
            candidate_sequence=candidate_sequence,
            old_sequence=old_sequence,
            old_cursor=old_cursor,
            old_tail_steps_used=max(0, old_cursor - self.chunk_size),
            selection=selection,
            observation_age_ns=observation_age_ns,
            standby_wait_ns=standby_wait_ns,
            timing_rate_hz=timing_rate_hz,
        )

    def advance_after_send(self) -> None:
        if self.exhausted:
            raise RuntimeError("cannot advance an exhausted action horizon")
        self.cursor += 1

    @staticmethod
    def _validate_chunk_timing(chunk: SchedulableChunk, label: str) -> None:
        sequence = _require_plain_int(f"{label}.sequence", chunk.sequence)
        if sequence <= 0:
            raise ValueError(f"{label}.sequence must be positive")
        observation_ns = _require_plain_int(
            f"{label}.observation_monotonic_ns",
            chunk.observation_monotonic_ns,
        )
        if observation_ns <= 0:
            raise ValueError(
                f"{label}.observation_monotonic_ns must be positive"
            )
        _validated_policy_rate(chunk.policy_rate_hz)
        validated_execution_indices(chunk.execution_indices)


def _require_plain_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    return value


def _validated_policy_rate(policy_rate_hz: float) -> Fraction:
    if isinstance(policy_rate_hz, bool) or not isinstance(
        policy_rate_hz, (int, float)
    ):
        raise ValueError(
            "policy_rate_hz must be a positive finite number; "
            f"got {policy_rate_hz!r}"
        )
    numeric_rate = float(policy_rate_hz)
    if not math.isfinite(numeric_rate) or numeric_rate <= 0.0:
        raise ValueError(
            "policy_rate_hz must be a positive finite number; "
            f"got {policy_rate_hz!r}"
        )
    # Fraction(str(...)) preserves operator-facing decimal rates without the
    # binary floating-point boundary drift that would turn an exact 100 ms
    # deadline into an expired target.
    return Fraction(str(policy_rate_hz))


def effective_action_timing_rate_hz(
    *,
    control_rate_hz: float,
    policy_rate_hz: float,
    schedule_explicit: bool,
) -> float:
    """Return the raw-index clock used for replacement time alignment.

    An explicit schedule may subsample raw policy targets (for example raw
    30 Hz indices 2,5,8 at a 10 Hz robot rate), so raw target timestamps keep
    the policy rate.  A legacy consecutive H50 schedule is intentionally
    re-timed when the operator selects 10/15/20/25/30 Hz, so it uses the
    selected robot control rate.
    """

    control_rate = float(_validated_policy_rate(control_rate_hz))
    policy_rate = float(_validated_policy_rate(policy_rate_hz))
    if not isinstance(schedule_explicit, bool):
        raise ValueError(
            "schedule_explicit must be bool; "
            f"got {schedule_explicit!r}"
        )
    return policy_rate if schedule_explicit else control_rate


def advance_periodic_deadline(
    *, previous_deadline: float, period_seconds: float, finished_at: float
) -> PeriodicDeadlineUpdate:
    """Advance one periodic send slot without issuing catch-up bursts."""

    values = {
        "previous_deadline": previous_deadline,
        "period_seconds": period_seconds,
        "finished_at": finished_at,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number; got {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite; got {value!r}")
    previous = float(previous_deadline)
    period = float(period_seconds)
    finished = float(finished_at)
    if previous < 0.0 or finished < 0.0 or period <= 0.0:
        raise ValueError(
            "deadlines must be non-negative and period_seconds must be positive"
        )
    scheduled = previous + period
    if scheduled <= finished:
        return PeriodicDeadlineUpdate(
            next_deadline=finished + period,
            overrun=True,
            lateness_seconds=finished - scheduled,
        )
    return PeriodicDeadlineUpdate(
        next_deadline=scheduled,
        overrun=False,
        lateness_seconds=0.0,
    )


def wall_time_to_monotonic_ns(
    *,
    observation_wall_time_ns: int,
    sampled_wall_time_ns: int,
    sampled_monotonic_ns: int,
    max_future_wall_time_ns: int = DEFAULT_MAX_FUTURE_WALL_TIME_NS,
) -> int:
    """Map a same-host wall timestamp to monotonic time conservatively.

    Small positive source-clock jitter is clamped to the sampling instant so
    it cannot make a future observation look newer than it is. A timestamp
    farther in the future than ``max_future_wall_time_ns`` is rejected.
    """

    observation_ns = _require_plain_int(
        "observation_wall_time_ns", observation_wall_time_ns
    )
    sampled_wall_ns = _require_plain_int(
        "sampled_wall_time_ns", sampled_wall_time_ns
    )
    sampled_monotonic = _require_plain_int(
        "sampled_monotonic_ns", sampled_monotonic_ns
    )
    max_future_ns = _require_plain_int(
        "max_future_wall_time_ns", max_future_wall_time_ns
    )
    if observation_ns <= 0 or sampled_wall_ns <= 0 or sampled_monotonic <= 0:
        raise ValueError("wall and monotonic timestamps must be positive")
    if max_future_ns < 0:
        raise ValueError("max_future_wall_time_ns must be non-negative")

    wall_age_ns = sampled_wall_ns - observation_ns
    if wall_age_ns < -max_future_ns:
        raise ValueError(
            "observation wall timestamp is too far in the future: "
            f"future_ms={-wall_age_ns / 1e6:.3f} "
            f"limit_ms={max_future_ns / 1e6:.3f}"
        )
    mapped_ns = sampled_monotonic - max(0, wall_age_ns)
    if mapped_ns <= 0:
        raise ValueError(
            "observation wall timestamp cannot be represented on the sampled "
            "monotonic timeline"
        )
    return mapped_ns


def validated_execution_indices(
    execution_indices: Sequence[int],
) -> tuple[int, ...]:
    """Return a non-empty, strictly increasing raw-index schedule."""

    indices = tuple(execution_indices)
    if not indices:
        raise ValueError("execution_indices must not be empty")
    previous = -1
    for position, raw_index in enumerate(indices):
        _require_plain_int(f"execution_indices[{position}]", raw_index)
        if raw_index < 0:
            raise ValueError(
                f"execution_indices[{position}] must be non-negative"
            )
        if raw_index <= previous:
            raise ValueError(
                "execution_indices must be strictly increasing; "
                f"position={position} previous={previous} current={raw_index}"
            )
        previous = raw_index
    return indices


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def select_first_unexpired_action(
    *,
    policy_rate_hz: float,
    execution_indices: Sequence[int],
    observation_monotonic_ns: int,
    next_send_deadline_ns: int,
    start_position: int = 0,
) -> ExecutionSelection | None:
    """Select the first scheduled target at or after the next send deadline.

    ``start_position`` is a monotonic cursor: an earlier raw action is never
    selected even when the supplied deadline predates the observation anchor.
    Equality is not expired.  For example, at 10 Hz raw action 2 remains valid
    at exactly ``anchor + 300 ms`` and expires one nanosecond later.
    """

    rate = _validated_policy_rate(policy_rate_hz)
    indices = validated_execution_indices(execution_indices)
    observation_ns = _require_plain_int(
        "observation_monotonic_ns", observation_monotonic_ns
    )
    deadline_ns = _require_plain_int(
        "next_send_deadline_ns", next_send_deadline_ns
    )
    position = _require_plain_int("start_position", start_position)
    if not 0 <= position <= len(indices):
        raise ValueError(
            "start_position must be within the execution schedule; "
            f"got {position}, length={len(indices)}"
        )

    for execution_position in range(position, len(indices)):
        raw_policy_index = indices[execution_position]
        target_offset_ns = (
            Fraction((raw_policy_index + 1) * NANOSECONDS_PER_SECOND, 1)
            / rate
        )
        target_ns = Fraction(observation_ns, 1) + target_offset_ns
        if target_ns >= deadline_ns:
            return ExecutionSelection(
                execution_position=execution_position,
                raw_policy_index=raw_policy_index,
                target_monotonic_ns=_ceil_fraction(target_ns),
            )
    return None


def plan_chunk_prefix(
    *,
    execution_indices: Sequence[int],
    chunk_size: int,
    lead_steps: int,
    start_position: int,
) -> ChunkPrefixPlan:
    """Plan prefetch and exhaustion positions for a fixed chunk prefix.

    Prefetch becomes due before sending ``prefetch_position``.  With
    ``chunk_size=40`` and ``lead_steps=5``, positions 35..39 form the old
    chunk's five-step asynchronous tail and position 40 is the exclusive
    planned boundary.
    """

    indices = validated_execution_indices(execution_indices)
    size = _require_plain_int("chunk_size", chunk_size)
    lead = _require_plain_int("lead_steps", lead_steps)
    start = _require_plain_int("start_position", start_position)
    if not 1 <= size <= len(indices):
        raise ValueError(
            "chunk_size must be within the executable schedule; "
            f"got {size}, length={len(indices)}"
        )
    if not 1 <= lead < size:
        raise ValueError(
            "lead_steps must satisfy 1 <= lead_steps < chunk_size; "
            f"got lead_steps={lead}, chunk_size={size}"
        )
    if not 0 <= start <= len(indices):
        raise ValueError(
            "start_position must be within the execution schedule; "
            f"got {start}, length={len(indices)}"
        )

    boundary = size
    prefetch = boundary - lead
    usable_start = min(start, boundary)
    positions = tuple(range(usable_start, boundary))
    raw_indices = tuple(indices[position] for position in positions)
    return ChunkPrefixPlan(
        start_position=start,
        prefetch_position=prefetch,
        planned_boundary_position=boundary,
        usable_tail_positions=positions,
        usable_tail_raw_indices=raw_indices,
        steps_until_prefetch=max(0, prefetch - start),
        usable_tail_steps=len(positions),
        tail_steps_at_prefetch=max(0, boundary - max(start, prefetch)),
    )
