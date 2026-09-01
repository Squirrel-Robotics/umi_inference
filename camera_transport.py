#!/usr/bin/env python3
"""Compact, fail-closed transport for timestamped ROS wrist images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


WRIST_TRANSPORT_SCHEMA = "umi_ros_compressed_wrist_v2"
WRIST_TRANSPORT_TRANSFORM = "decode_bgr_to_rgb_no_rotation"
TRANSPORT_PROBE_SCHEMA = "umi_transport_probe_v1"
EXPECTED_WRIST_SHAPE = (480, 640, 3)
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_TRANSPORT_PROBE_BYTES = 16 * 1024


def _as_bytes(encoded: Any) -> bytes:
    try:
        blob = bytes(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("compressed wrist image is not bytes-like") from exc
    if not blob or len(blob) > MAX_COMPRESSED_BYTES:
        raise ValueError(
            f"compressed wrist image size is invalid: {len(blob)} bytes"
        )
    return blob


def _detect_codec(blob: bytes, format_hint: str = "") -> str:
    hint = str(format_hint).lower()
    if "jpeg" in hint or "jpg" in hint or blob.startswith(b"\xff\xd8"):
        return "jpeg"
    if "png" in hint or blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    raise ValueError(f"unsupported compressed wrist format: {format_hint!r}")


def decode_compressed_wrist_rgb(encoded: Any) -> np.ndarray:
    """Decode a ROS CompressedImage without rotating or mirroring it."""

    blob = _as_bytes(encoded)
    image_bgr = cv2.imdecode(
        np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image_bgr is None:
        raise ValueError("OpenCV could not decode compressed wrist image")
    image_rgb = np.ascontiguousarray(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )
    if image_rgb.shape != EXPECTED_WRIST_SHAPE:
        raise ValueError(
            "unexpected decoded wrist shape: "
            f"{image_rgb.shape}, expected {EXPECTED_WRIST_SHAPE}"
        )
    return image_rgb


def make_wrist_transport_payload(
    encoded: Any,
    *,
    format_hint: str,
    decoded_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Wrap the original ROS payload without recompressing or relabeling it."""

    blob = _as_bytes(encoded)
    if tuple(decoded_shape) != EXPECTED_WRIST_SHAPE:
        raise ValueError(
            f"unexpected wrist shape: {decoded_shape}, expected {EXPECTED_WRIST_SHAPE}"
        )
    return {
        "schema": WRIST_TRANSPORT_SCHEMA,
        "codec": _detect_codec(blob, format_hint),
        "transform": WRIST_TRANSPORT_TRANSFORM,
        "decoded_shape": list(EXPECTED_WRIST_SHAPE),
        "data": blob,
    }


def decode_wrist_transport(value: Any) -> np.ndarray:
    """Decode the compact payload; accept old raw RGB only for diagnostics."""

    if isinstance(value, np.ndarray):
        image = np.asarray(value)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[-1] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError(
                "raw wrist RGB has an invalid contract: "
                f"dtype={image.dtype} shape={image.shape}"
            )
        return np.ascontiguousarray(image)
    if not isinstance(value, dict):
        raise ValueError("wrist image transport payload must be a mapping")
    if value.get("schema") != WRIST_TRANSPORT_SCHEMA:
        raise ValueError(f"unexpected wrist transport schema: {value.get('schema')!r}")
    if value.get("transform") != WRIST_TRANSPORT_TRANSFORM:
        raise ValueError(
            f"unexpected wrist transport transform: {value.get('transform')!r}"
        )
    expected_shape = value.get("decoded_shape")
    if expected_shape != list(EXPECTED_WRIST_SHAPE):
        raise ValueError(f"unexpected advertised wrist shape: {expected_shape!r}")
    blob = _as_bytes(value.get("data"))
    codec = _detect_codec(blob, str(value.get("codec", "")))
    if codec != value.get("codec"):
        raise ValueError(
            f"wrist codec does not match encoded bytes: {value.get('codec')!r}"
        )
    return decode_compressed_wrist_rgb(blob)
