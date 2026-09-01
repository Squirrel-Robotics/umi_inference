#!/usr/bin/env python3
"""Publish the two XR wrist UVC cameras without any robot-control interfaces."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import signal
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


CAMERAS = (
    ("left", "/dev/ArmCamLeft", "/camera1/usb_cam1/image_raw/image_compressed"),
    ("right", "/dev/ArmCamRight", "/camera3/usb_cam3/image_raw/image_compressed"),
)

IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class DuplicatePublisherDetected(Exception):
    """A vendor publisher appeared; exit successfully so systemd will not restart."""


def open_camera(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {device}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
    # These fisheye cameras expose MJPG 640x480 only at 60 fps even when 30
    # fps is requested. A faster capture mode is valid: run() is the single
    # place that rate-limits publication to exactly 30 Hz. Reject only a
    # source that cannot sustain the required output rate.
    if (width, height) != (640, 480) or fps < 29.0 or fourcc != "MJPG":
        cap.release()
        raise RuntimeError(
            f"{device} rejected required MJPG 640x480 at >=30 fps "
            f"(reported {fourcc} {width}x{height}@{fps:.3f})"
        )
    return cap


class WristCameraPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pi05_wrist_camera_publisher")
        self._captures: list[tuple[str, cv2.VideoCapture]] = []
        self._executor: ThreadPoolExecutor | None = None
        # Node owns an internal ``_publishers`` list.  Keep the application
        # handles separate: create_publisher() already appends to Node's list,
        # and reusing that name would duplicate every handle and also corrupt
        # Node.destroy_node().
        self._image_publishers = []
        # On a systemd retry, a vendor source may have appeared since the
        # bridge made its initial decision. Check before touching either UVC.
        rclpy.spin_once(self, timeout_sec=0.25)
        for _, _, topic in CAMERAS:
            if self.count_publishers(topic) > 0:
                raise DuplicatePublisherDetected(f"existing publisher detected on {topic}")
        try:
            for name, device, topic in CAMERAS:
                self._captures.append((name, open_camera(device)))
                self._image_publishers.append(
                    self.create_publisher(CompressedImage, topic, IMAGE_QOS)
                )
            # Each UVC handle has one persistent worker lane per publication
            # cycle.  publish_once() never submits the next read until both
            # current reads have completed, so a VideoCapture is never entered
            # concurrently while the two independent cameras still run in
            # parallel.
            self._executor = ThreadPoolExecutor(
                max_workers=len(self._captures),
                thread_name_prefix="wrist-camera",
            )
        except Exception:
            self.close()
            raise
        self._period = 1.0 / 30.0
        self._next_frame = time.monotonic()
        self._next_graph_check = self._next_frame + 1.0

    def _capture_and_encode(
        self,
        name: str,
        device: str,
        cap: cv2.VideoCapture,
    ) -> tuple[object, bytes]:
        """Read and encode one camera entirely inside its dedicated work item."""
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"frame read failed for {name} wrist camera ({device})")
        # This is the real completion time of this camera's acquisition, not a
        # shared time generated after both cameras finish and not the publish
        # time.  JPEG work happens only after the stamp has been captured.
        stamp = self.get_clock().now().to_msg()
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {name} wrist camera")
        return stamp, encoded.tobytes()

    def publish_once(self) -> None:
        executor = self._executor
        if executor is None:
            raise RuntimeError("wrist camera workers are not available")

        futures = [
            executor.submit(self._capture_and_encode, name, device, cap)
            for (name, cap), (_, device, _) in zip(self._captures, CAMERAS, strict=True)
        ]
        # Always wait for both jobs before inspecting either exception.  This
        # guarantees that an error on one camera cannot leave the other job
        # accessing its VideoCapture while close() releases the handles.
        wait(futures)
        frames = [future.result() for future in futures]

        # No rotation or mirroring is applied.
        for publisher, (stamp, data) in zip(self._image_publishers, frames, strict=True):
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = ""
            msg.format = "jpeg"
            msg.data = data
            publisher.publish(msg)

    def run(self) -> None:
        while rclpy.ok():
            now = time.monotonic()
            if now >= self._next_graph_check:
                for _, _, topic in CAMERAS:
                    if self.count_publishers(topic) != 1:
                        raise DuplicatePublisherDetected(
                            f"another publisher appeared on {topic}; stopping both independent cameras"
                        )
                self._next_graph_check = now + 1.0
            self.publish_once()
            rclpy.spin_once(self, timeout_sec=0.0)
            self._next_frame += self._period
            delay = self._next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_frame = time.monotonic()

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            # No VideoCapture is released while a worker may still be inside
            # cap.read() or cv2.imencode().
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for _, cap in getattr(self, "_captures", []):
            cap.release()
        self._captures = []


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rclpy.init()
    node = None
    try:
        node = WristCameraPublisher()
        node.run()
        return 0
    except DuplicatePublisherDetected as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: rclpy.shutdown())
    raise SystemExit(main())
