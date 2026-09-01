#!/usr/bin/env python3
"""Publish Revo2 motor positions without creating any command subscriptions."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from typing import Any

sys.path.insert(0, "/bridge")
import vr_trigger_revo2_bridge as bridge  # noqa: E402


LOGGER = logging.getLogger("revo2_feedback_only")


def start_runtime(specs: tuple[Any, ...]) -> Any:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("pi05_revo2_feedback_only")
    qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    publishers = {
        ("state", spec.side): node.create_publisher(String, f"/revo2/state/{spec.side}", qos)
        for spec in specs
    }
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, name="revo2-feedback-ros", daemon=True)
    thread.start()
    return bridge.RosRuntime(
        rclpy=rclpy,
        node=node,
        executor=executor,
        thread=thread,
        publishers=publishers,
        string_message_type=String,
    )


async def run(
    rate: float,
    timeout: float,
    max_consecutive_errors: int,
    stop: threading.Event,
    left_port: str | None,
    right_port: str | None,
) -> None:
    from bc_stark_sdk import main_mod as sdk

    # This process has no command subscriptions and never sends a position
    # target. After connect_hands verifies each serial number and hand type, it
    # is therefore safe to restore the deployment's Normalized unit contract
    # if a hand power cycle returned it to Physical mode.
    bridge_args = [
        "--hands",
        "both",
        "--input-source",
        "absolute",
        "--set-normalized-mode",
    ]
    if left_port:
        bridge_args.extend(["--left-port", left_port])
    if right_port:
        bridge_args.extend(["--right-port", right_port])
    args = bridge.parse_args(bridge_args)
    specs = bridge.selected_hand_specs(args)
    runtime = start_runtime(specs)
    connected: dict[str, Any] = {}
    try:
        connected = await bridge.connect_hands(args, sdk, specs)
        sdk_version = bridge.sdk_version_string(sdk)
        interval = 1.0 / rate
        consecutive_errors = {side: 0 for side in connected}
        LOGGER.info("read-only feedback active at %.1f Hz; no command subscriptions exist", rate)
        while not stop.is_set():
            started = time.monotonic()

            async def sample(
                side: str, hand: Any
            ) -> tuple[str, Any | None, Exception | None]:
                try:
                    async with hand.lock:
                        status = await asyncio.wait_for(
                            hand.client.get_motor_status(hand.spec.slave_id),
                            timeout=timeout,
                        )
                except Exception as exc:
                    return side, None, exc
                return side, status, None

            results = await asyncio.gather(
                *(sample(side, hand) for side, hand in connected.items())
            )
            fatal_error: tuple[str, Exception, int] | None = None
            for side, status, error in results:
                if error is not None:
                    consecutive_errors[side] += 1
                    LOGGER.warning(
                        "%s feedback read failed (%d/%d): %s: %s",
                        side,
                        consecutive_errors[side],
                        max_consecutive_errors,
                        type(error).__name__,
                        error,
                    )
                    if consecutive_errors[side] >= max_consecutive_errors:
                        fatal_error = (
                            side,
                            error,
                            consecutive_errors[side],
                        )
                    continue
                if consecutive_errors[side]:
                    LOGGER.info("%s feedback stream recovered", side)
                consecutive_errors[side] = 0
                runtime.publish(
                    "state",
                    side,
                    bridge.state_payload(side, status, time.time(), sdk_version),
                )
            if fatal_error is not None:
                side, error, count = fatal_error
                raise RuntimeError(
                    f"{side} feedback failed {count} consecutive times"
                ) from error
            await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        for hand in connected.values():
            await bridge.close_client(sdk, hand.client)
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument("--left-port")
    parser.add_argument("--right-port")
    args = parser.parse_args()
    if not 1.0 <= args.rate <= 100.0:
        parser.error("--rate must be in the range 1..100 Hz")
    if not 0.05 <= args.timeout <= 5.0:
        parser.error("--timeout must be in the range 0.05..5 seconds")
    if not 1 <= args.max_consecutive_errors <= 100:
        parser.error("--max-consecutive-errors must be in the range 1..100")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        asyncio.run(
            run(
                args.rate,
                args.timeout,
                args.max_consecutive_errors,
                stop,
                args.left_port,
                args.right_port,
            )
        )
    except Exception as exc:
        LOGGER.error("feedback bridge failed: %s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
