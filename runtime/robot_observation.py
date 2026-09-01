#!/usr/bin/env python3
"""Timestamp-aligned ROS observations used by the live UMI controller.

Only timestamped ROS sources are accepted here. Wrist frames keep their
original compressed bytes for transport; decoding is performed once on XR to
validate the no-rotation RGB transform and image shape.
"""

from __future__ import annotations

from collections import deque
import json
import threading
import time
from typing import Any, Callable

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from camera_sync import (
    DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    TimedCameraFrame,
    select_latest_aligned_pair,
    select_nearest_timed_sample,
)
from camera_transport import (
    decode_compressed_wrist_rgb,
    make_wrist_transport_payload,
)


# The current robot's physical wrist views are carried on the opposite vendor
# topic numbers: camera3 is the left-hand view and camera1 is the right-hand
# view.  Keep the semantic model keys correct here; do not rotate, mirror, or
# exchange them again on the policy server.
LEFT_CAMERA = "/camera3/usb_cam3/image_raw/image_compressed"
RIGHT_CAMERA = "/camera1/usb_cam1/image_raw/image_compressed"
# The current XR stack publishes the live wrist FK on the whole-body
# controller feedback topics.  The legacy /left|right_arm/state/end_pose
# names may remain discoverable without a publisher, so subscribing to them
# makes a live snapshot wait forever.
LEFT_EEF = "/whole_body_controller/left_wrist_pose"
RIGHT_EEF = "/whole_body_controller/right_wrist_pose"
LEFT_HAND = "/revo2/state/left"
RIGHT_HAND = "/revo2/state/right"

CAMERA_KEYS = ("cam_left_wrist", "cam_right_wrist")
ROBOT_KEYS = ("left_eef", "right_eef", "left_hand", "right_hand")
HAND_NAMES = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
HAND_MAX_DEG = np.array([59, 90, 81, 81, 81, 81], dtype=np.float32)
DEFAULT_CAMERA_SYNC_MAX_SKEW_MS = 50.0
CAMERA_HISTORY_FRAMES = 64
ROBOT_STATE_HISTORY_FRAMES = 1024


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array(
        [
            [1 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1 - xx - yy],
        ],
        dtype=np.float64,
    )


class ObservationNode(Node):
    """Collect both wrist cameras and robot feedback in one XR clock domain."""

    def __init__(
        self,
        *,
        camera_sync_max_skew_ms: float = DEFAULT_CAMERA_SYNC_MAX_SKEW_MS,
    ) -> None:
        super().__init__("umi_pi05_live_observation")
        if not np.isfinite(camera_sync_max_skew_ms) or camera_sync_max_skew_ms <= 0:
            raise ValueError("camera_sync_max_skew_ms must be positive")

        self.lock = threading.Lock()
        self.values: dict[str, tuple[float, np.ndarray]] = {}
        self.camera_histories: dict[str, deque[TimedCameraFrame]] = {
            key: deque(maxlen=CAMERA_HISTORY_FRAMES) for key in CAMERA_KEYS
        }
        self.robot_histories: dict[str, deque[TimedCameraFrame]] = {
            key: deque(maxlen=ROBOT_STATE_HISTORY_FRAMES) for key in ROBOT_KEYS
        }
        self._sequences = {key: 0 for key in (*CAMERA_KEYS, *ROBOT_KEYS)}
        self.camera_sync_max_skew_ms = float(camera_sync_max_skew_ms)

        camera_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            CompressedImage,
            LEFT_CAMERA,
            lambda message: self._image("cam_left_wrist", message),
            camera_qos,
        )
        self.create_subscription(
            CompressedImage,
            RIGHT_CAMERA,
            lambda message: self._image("cam_right_wrist", message),
            camera_qos,
        )
        self.create_subscription(
            PoseStamped,
            LEFT_EEF,
            lambda message: self._pose("left_eef", message),
            state_qos,
        )
        self.create_subscription(
            PoseStamped,
            RIGHT_EEF,
            lambda message: self._pose("right_eef", message),
            state_qos,
        )
        self.create_subscription(
            String,
            LEFT_HAND,
            lambda message: self._hand("left_hand", message),
            state_qos,
        )
        self.create_subscription(
            String,
            RIGHT_HAND,
            lambda message: self._hand("right_hand", message),
            state_qos,
        )

    def _store(
        self,
        key: str,
        value: Any,
        *,
        source_wall_time_ns: int,
        source: str,
        timestamp_semantics: str,
        arrival_wall_time_ns: int | None = None,
        arrival_monotonic_ns: int | None = None,
        transport_value: Any | None = None,
    ) -> None:
        if source_wall_time_ns <= 0:
            raise ValueError(f"{key} source timestamp must be positive")
        arrival_wall_time_ns = time.time_ns() if arrival_wall_time_ns is None else arrival_wall_time_ns
        arrival_monotonic_ns = (
            time.monotonic_ns()
            if arrival_monotonic_ns is None
            else arrival_monotonic_ns
        )
        with self.lock:
            self._sequences[key] += 1
            sample = TimedCameraFrame(
                value=value,
                source_wall_time_ns=int(source_wall_time_ns),
                arrival_wall_time_ns=int(arrival_wall_time_ns),
                arrival_monotonic_ns=int(arrival_monotonic_ns),
                sequence=self._sequences[key],
                source=source,
                timestamp_semantics=timestamp_semantics,
                transport_value=transport_value,
            )
            histories = (
                self.camera_histories if key in self.camera_histories else self.robot_histories
            )
            histories[key].append(sample)
            if key in self.robot_histories:
                self.values[key] = (arrival_monotonic_ns / 1e9, np.asarray(value))

    @staticmethod
    def _ros_stamp_ns(message: Any) -> int:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _image(self, key: str, message: CompressedImage) -> None:
        arrival_wall_time_ns = time.time_ns()
        arrival_monotonic_ns = time.monotonic_ns()
        source_wall_time_ns = self._ros_stamp_ns(message)
        if source_wall_time_ns <= 0:
            return
        try:
            decoded_shape = decode_compressed_wrist_rgb(message.data).shape
            transport = make_wrist_transport_payload(
                message.data,
                format_hint=message.format,
                decoded_shape=decoded_shape,
            )
        except ValueError:
            return
        self._store(
            key,
            transport,
            source_wall_time_ns=source_wall_time_ns,
            arrival_wall_time_ns=arrival_wall_time_ns,
            arrival_monotonic_ns=arrival_monotonic_ns,
            source=LEFT_CAMERA if key == "cam_left_wrist" else RIGHT_CAMERA,
            timestamp_semantics="ros_header_stamp",
            transport_value=transport,
        )

    def _pose(self, key: str, message: PoseStamped) -> None:
        source_wall_time_ns = self._ros_stamp_ns(message)
        if source_wall_time_ns <= 0:
            return
        position = message.pose.position
        quaternion = message.pose.orientation
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_matrix(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        transform[:3, 3] = [position.x, position.y, position.z]
        self._store(
            key,
            transform,
            source_wall_time_ns=source_wall_time_ns,
            source=LEFT_EEF if key == "left_eef" else RIGHT_EEF,
            timestamp_semantics="ros_header_stamp",
        )

    def _hand(self, key: str, message: String) -> None:
        payload = json.loads(message.data)
        if payload.get("schema") != "revo2_joint_state_v1":
            raise ValueError(f"unexpected Revo2 schema: {payload.get('schema')}")
        if payload.get("actuator_names") != HAND_NAMES:
            raise ValueError(
                f"unexpected Revo2 actuator order: {payload.get('actuator_names')}"
            )
        positions = np.asarray(payload["positions"], dtype=np.float32)
        if positions.shape != (6,) or np.any(positions < 0) or np.any(positions > 1000):
            raise ValueError(f"invalid Revo2 positions: {positions}")
        timestamp = payload.get("timestamp")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not np.isfinite(timestamp)
            or timestamp <= 0
        ):
            raise ValueError(f"invalid Revo2 timestamp: {timestamp!r}")
        self._store(
            key,
            positions / 1000.0 * HAND_MAX_DEG,
            source_wall_time_ns=int(round(float(timestamp) * 1e9)),
            source=LEFT_HAND if key == "left_hand" else RIGHT_HAND,
            timestamp_semantics="revo2_payload_timestamp",
        )

    def _select_wrist_pair_locked(
        self,
        *,
        now_monotonic_ns: int,
        max_age: float,
        after_sync: dict[str, Any] | None,
    ) -> tuple[TimedCameraFrame, TimedCameraFrame, dict[str, Any]]:
        after_observation_ns = None
        after_sequences = None
        if after_sync is not None:
            after_observation_ns = int(after_sync["observation_wall_time_ns"])
            cameras = after_sync["cameras"]
            after_sequences = {
                key: int(cameras[key]["sequence"]) for key in CAMERA_KEYS
            }
        left, right, sync = select_latest_aligned_pair(
            tuple(self.camera_histories["cam_left_wrist"]),
            tuple(self.camera_histories["cam_right_wrist"]),
            now_monotonic_ns=now_monotonic_ns,
            max_age_ns=int(max_age * 1e9),
            max_skew_ns=int(self.camera_sync_max_skew_ms * 1e6),
            after_observation_wall_time_ns=after_observation_ns,
            after_sequences=after_sequences,
        )
        sync["left_camera_owner"] = "ros"
        return left, right, sync

    def snapshot(
        self,
        max_age: float,
        *,
        after_sync: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        now_monotonic_ns = time.monotonic_ns()
        now = now_monotonic_ns / 1e9
        with self.lock:
            missing = [key for key in ROBOT_KEYS if key not in self.values]
            missing += [key for key in CAMERA_KEYS if not self.camera_histories[key]]
            if missing:
                return None, "missing=" + ",".join(missing)
            stale = [
                key for key in ROBOT_KEYS if now - self.values[key][0] > max_age
            ]
            if stale:
                return None, "stale=" + ",".join(stale)
            try:
                left, right, sync = self._select_wrist_pair_locked(
                    now_monotonic_ns=now_monotonic_ns,
                    max_age=max_age,
                    after_sync=after_sync,
                )
            except RuntimeError as exc:
                return None, str(exc)

            observation_ns = int(sync["observation_wall_time_ns"])
            selected: dict[str, TimedCameraFrame] = {}
            state_timing: dict[str, Any] = {}
            for key in ROBOT_KEYS:
                try:
                    sample, metadata = select_nearest_timed_sample(
                        tuple(self.robot_histories[key]),
                        target_wall_time_ns=observation_ns,
                        now_monotonic_ns=now_monotonic_ns,
                        max_age_ns=int(max_age * 1e9),
                        max_skew_ns=int(DEFAULT_STATE_SYNC_MAX_SKEW_MS * 1e6),
                    )
                except RuntimeError as exc:
                    return None, f"{key}: {exc}"
                selected[key] = sample
                state_timing[key] = metadata

            return {
                **{key: np.asarray(sample.value).copy() for key, sample in selected.items()},
                "_sync": sync,
                "_state_timing": state_timing,
                "_transport_images": {
                    "cam_left_wrist": left.transport_value,
                    "cam_right_wrist": right.transport_value,
                },
            }, "ready"


def wrist_images_for_transport(snapshot: dict[str, Any]) -> dict[str, Any]:
    images = snapshot.get("_transport_images")
    if not isinstance(images, dict) or any(key not in images for key in CAMERA_KEYS):
        raise RuntimeError("aligned observation has no compressed wrist payloads")
    return {key: images[key] for key in CAMERA_KEYS}


def wait_snapshot(
    node: ObservationNode,
    timeout: float,
    max_age: float,
    *,
    after_sync: dict[str, Any] | None = None,
    stop_check: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    reason = "no data"
    while time.monotonic() < deadline:
        if stop_check is not None:
            stop_check("robot observation wait")
        snapshot, reason = node.snapshot(max_age, after_sync=after_sync)
        if snapshot is not None:
            if stop_check is not None:
                stop_check("robot observation ready")
            return snapshot
        time.sleep(0.05)
    raise RuntimeError(f"robot observation not ready after {timeout:.1f}s: {reason}")
