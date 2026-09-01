#!/usr/bin/env python3
"""Pure, no-ROS cooperative stop primitives for live robot inference."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path
import signal
import threading
import time
from typing import Any


class StopRequested(BaseException):
    """Control-flow exception raised by explicit execution-boundary checks."""


class ExecutionGate:
    """One gate shared by startup, inference and every hardware side effect."""

    def __init__(
        self,
        stop_event: threading.Event,
        *,
        enable_file: Path,
        execute: bool,
    ) -> None:
        self.stop_event = stop_event
        self.enable_file = enable_file
        self.execute = execute
        # A Python signal handler runs on the main thread between bytecodes.
        # RLock prevents a signal that lands while that same thread is reading
        # gate state from deadlocking while it closes the gate.
        self._lock = threading.RLock()
        self._requested = False
        self._reason: str | None = None
        self._signal_number: int | None = None
        self._sigint_count = 0
        self._force_emergency = False
        self._live_output_enabled = False

    @property
    def requested(self) -> bool:
        with self._lock:
            return self._requested

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def signal_number(self) -> int | None:
        with self._lock:
            return self._signal_number

    @property
    def force_emergency(self) -> bool:
        with self._lock:
            return self._force_emergency

    @property
    def live_output_enabled(self) -> bool:
        with self._lock:
            return self._live_output_enabled

    def mark_live_output_enabled(self) -> None:
        self.check("mark live output enabled")
        with self._lock:
            self._live_output_enabled = True

    def request_stop(
        self,
        reason: str,
        *,
        signal_number: int | None = None,
        force_emergency: bool = False,
    ) -> None:
        with self._lock:
            first = not self._requested
            self._requested = True
            if first:
                self._reason = reason
            if signal_number is not None and self._signal_number is None:
                self._signal_number = int(signal_number)
            self._force_emergency = self._force_emergency or force_emergency
        self.stop_event.set()
        if self.execute:
            try:
                self.enable_file.unlink(missing_ok=True)
            except OSError:
                # The shell EXIT trap is an independent cleanup layer.  The
                # in-process gate remains closed even if unlink itself fails.
                pass

    def handle_signal(self, signum: int, _frame: Any) -> None:
        # Never raise asynchronously from a Python signal handler: it could
        # split a two-sided hardware dispatch or interrupt a library while it
        # owns internal state. All blocking waits poll this event, and every
        # hardware boundary calls check(), which raises StopRequested there.
        with self._lock:
            # Cooperative stop (for example LIVE_ENABLE removal) isn't a
            # signal. The first actual SIGINT must remain a soft stop even if
            # the shell removed the enable file a few microseconds earlier.
            if signum == signal.SIGINT:
                self._sigint_count += 1
            self._requested = True
            if self._reason is None:
                self._reason = f"signal {signal.Signals(signum).name}"
            if self._signal_number is None:
                self._signal_number = int(signum)
            # Only a second explicit Ctrl+C requests robot emergency stop.
            # A wrapper watchdog uses SIGTERM; it must terminate a stuck
            # process without converting one user Ctrl+C into a robot e-stop.
            self._force_emergency = (
                self._force_emergency or self._sigint_count >= 2
            )
        # Do not call Event.set() from the signal handler. Event owns a
        # non-reentrant Condition internally, so an unlucky signal could
        # otherwise deadlock the main thread. All waits poll at <=50 ms and
        # observe _requested at the next explicit gate boundary.

    def check(self, stage: str) -> None:
        if self.execute and not self.enable_file.is_file() and not self.requested:
            self.request_stop(f"enable file removed before {stage}")
        if self.stop_event.is_set() or self.requested:
            raise StopRequested(self.reason or f"stop requested before {stage}")

    def wait(self, seconds: float, stage: str) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            self.check(stage)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            self.stop_event.wait(min(0.05, remaining))

    def wait_until(self, deadline: float, stage: str) -> None:
        self.wait(max(0.0, deadline - time.monotonic()), stage)


def wait_future_interruptibly(
    future: Future[Any],
    gate: ExecutionGate,
    stage: str,
    *,
    poll_seconds: float = 0.05,
) -> Any:
    """Wait for a worker without hiding stop requests behind Future.result()."""

    while True:
        gate.check(stage)
        try:
            value = future.result(timeout=poll_seconds)
        except FutureTimeoutError:
            # concurrent.futures.TimeoutError aliases builtin TimeoutError.
            # A completed worker can therefore *raise* TimeoutError (for
            # example a policy socket timeout). Do not mistake that real
            # failure for an unfinished Future and spin forever.
            if not future.done():
                continue
            value = future.result()
        gate.check(stage + " completed")
        return value
