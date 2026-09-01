#!/usr/bin/env python3
"""Independent X2 emergency stop used only after a controller stop timeout."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--reason", default="stuck_live_controller")
    args = parser.parse_args()
    if not 0.1 <= args.timeout <= 5.0:
        raise ValueError("--timeout must be between 0.1 and 5.0 seconds")

    from x2robot import connect

    robot = connect("x2://localhost:50051")
    result = robot.robot_control.emergency_stop(timeout=args.timeout)
    if result is None or not result.is_success:
        raise RuntimeError(
            "X2 emergency stop failed: "
            f"{getattr(result, 'error_message', '')}"
        )
    print(f"X2_EMERGENCY_STOP_CONFIRMED reason={args.reason}", flush=True)


if __name__ == "__main__":
    main()
