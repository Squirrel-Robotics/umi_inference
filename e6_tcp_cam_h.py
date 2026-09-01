#!/usr/bin/env python3
"""Decode E6 Stream Protocol V3 RGB packets into the latest ``cam_h`` frame.

``com.ssnwt.e6stream.debug`` listens on the headset loopback interface at TCP
port 8554.  The policy host reaches it through ``adb forward``.  This is a
strictly framed ``E6S3`` stream, not a naked Annex-B byte stream: every packet
has a 48-byte big-endian header and RGB frames carry 96 bytes of metadata before
their H.265 access unit.  Feeding those headers or metadata to FFmpeg corrupts
the decoder, so framing is validated before any media bytes reach PyAV.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Any

try:
    import av
except ModuleNotFoundError:  # Protocol/unit tests do not require FFmpeg.
    av = None  # type: ignore[assignment]
import cv2
import numpy as np

from cam_high_roi import CamHighRoiConfig, crop_cam_high_roi
from inference_recorder import InferenceRecorder
from camera_transport import (
    MAX_TRANSPORT_PROBE_BYTES,
    TRANSPORT_PROBE_SCHEMA,
    WRIST_TRANSPORT_SCHEMA,
    decode_wrist_transport,
)
from camera_sync import (
    CLOCK_DOMAIN,
    CLOCK_SYNC_SCHEMA,
    DEFAULT_STATE_SYNC_MAX_SKEW_MS,
    OBSERVATION_RETRY_SCHEMA,
    OBSERVATION_TRANSPORT_AGE_POLICY,
    SYNC_SCHEMA,
)
from server_camera_sync import (
    ClockSyncUncertainForAlignment,
    build_three_camera_sync_proof,
    parse_client_sync,
    select_best_timestamp,
)


LOGGER = logging.getLogger(__name__)
TRAINING_CAM_H_SIZE = (640, 480)  # width, height
# Keep at least half of the 50 ms deployment budget for measured camera skew.
# The measured uncertainty is subtracted again in ``infer`` below, preserving
# the total ``timestamp span + clock uncertainty <= 50 ms`` contract.
CLOCK_SYNC_MAX_UNCERTAINTY_MS = 25.0

# Frozen E6 Stream Protocol V3 wire contract. Keep these values in sync with
# ``protocol_v3.h`` from the E6 Linux handoff; all wire integers are big-endian.
E6V3_MAGIC = 0x45365333  # ASCII "E6S3"
E6V3_VERSION = 3
E6V3_HEADER = struct.Struct(">IHHIHHIQQIQ")
E6V3_SESSION_START = struct.Struct(">IIIIIIIIIIIIQQQ")
E6V3_RGB_METADATA = struct.Struct(">QIIQQ8x3f4f3f4f")
E6V3_AUDIO_METADATA_SIZE = 24
E6V3_SESSION_STOP_SIZE = 72
E6V3_MAX_PAYLOAD_SIZE = 9 * 1024 * 1024
E6V3_MAX_RGB_CONFIG_SIZE = 256 * 1024
E6V3_MAX_AUDIO_CONFIG_SIZE = 4 * 1024
E6V3_NO_FRAME_ID = 0xFFFFFFFF
E6V3_UNAVAILABLE_TIMESTAMP_NS = 0xFFFFFFFFFFFFFFFF

E6V3_SESSION_START_TYPE = 1
E6V3_RGB_CODEC_CONFIG = 2
E6V3_RGB_FRAME = 3
E6V3_SESSION_STOP = 4
E6V3_AUDIO_CODEC_CONFIG = 5
E6V3_AUDIO_FRAME = 6

E6V3_FLAG_KEY_FRAME = 1 << 0
E6V3_FLAG_POSE_MATCHED = 1 << 1
E6V3_FLAG_LEFT_ACTIVE = 1 << 2
E6V3_FLAG_RIGHT_ACTIVE = 1 << 3
E6V3_KNOWN_RGB_FLAGS = (
    E6V3_FLAG_KEY_FRAME
    | E6V3_FLAG_POSE_MATCHED
    | E6V3_FLAG_LEFT_ACTIVE
    | E6V3_FLAG_RIGHT_ACTIVE
)

E6V3_RGB_CODEC_FOURCC = 0x48323635  # H265
E6V3_AUDIO_CODEC_FOURCC = 0x41414320  # AAC + space
E6V3_CLOCK_BOOTTIME = 1
E6V3_AUDIO_REQUIRED = 1
E6V3_RGB_METADATA_SIZE = E6V3_RGB_METADATA.size

E6V3_TRANSPORT = "tcp_e6_stream_protocol_v3_h265"
E6V3_TIMESTAMP_SEMANTICS = (
    "e6_v3_exposure_midpoint_e6_realtime_plus_explicit_5090_minus_e6_offset"
)
E6V3_DECODE_TIMESTAMP_SEMANTICS = "5090_decode_complete_realtime"


class E6V3ProtocolError(ValueError):
    """The TCP peer violated the frozen E6 Stream Protocol V3 contract."""


@dataclass(frozen=True)
class E6V3Header:
    packet_type: int
    flags: int
    payload_size: int
    session_id: int
    sequence_number: int
    frame_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class E6V3Session:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    clock_anchor_boottime_ns: int
    clock_anchor_realtime_ns: int
    clock_anchor_uncertainty_ns: int

    def source_realtime_ns(self, source_boottime_ns: int) -> int:
        mapped = (
            self.clock_anchor_realtime_ns
            + int(source_boottime_ns)
            - self.clock_anchor_boottime_ns
        )
        if mapped <= 0:
            raise E6V3ProtocolError("invalid E6 BOOTTIME-to-REALTIME mapping")
        return mapped


@dataclass(frozen=True)
class E6V3RgbFrame:
    exposure_start_boottime_ns: int
    exposure_duration_ns: int
    gain: int
    controller_sample_boottime_ns: int
    controller_residual_ns: int
    access_unit: bytes


def parse_e6v3_header(data: bytes) -> E6V3Header:
    """Decode and validate one fixed-size Protocol V3 header."""

    if len(data) != E6V3_HEADER.size:
        raise E6V3ProtocolError(
            f"truncated Protocol V3 header: {len(data)}/{E6V3_HEADER.size} bytes"
        )
    (
        magic,
        version,
        packet_type,
        flags,
        header_size,
        reserved,
        payload_size,
        session_id,
        sequence_number,
        frame_id,
        timestamp_ns,
    ) = E6V3_HEADER.unpack(data)
    if magic != E6V3_MAGIC or version != E6V3_VERSION:
        raise E6V3ProtocolError(
            f"bad Protocol V3 identity: magic=0x{magic:08x} version={version}"
        )
    if packet_type not in range(E6V3_SESSION_START_TYPE, E6V3_AUDIO_FRAME + 1):
        raise E6V3ProtocolError(f"unknown Protocol V3 packet type: {packet_type}")
    if header_size != E6V3_HEADER.size or reserved != 0:
        raise E6V3ProtocolError("invalid Protocol V3 header size or reserved field")
    if payload_size > E6V3_MAX_PAYLOAD_SIZE:
        raise E6V3ProtocolError(
            f"Protocol V3 payload exceeds limit: {payload_size}"
        )
    if packet_type == E6V3_RGB_FRAME:
        if flags & ~E6V3_KNOWN_RGB_FLAGS:
            raise E6V3ProtocolError(f"RGB frame has unknown flags: 0x{flags:x}")
    elif flags:
        raise E6V3ProtocolError("non-RGB Protocol V3 packet has flags")

    expected_exact = {
        E6V3_SESSION_START_TYPE: E6V3_SESSION_START.size,
        E6V3_SESSION_STOP: E6V3_SESSION_STOP_SIZE,
    }
    if packet_type in expected_exact and payload_size != expected_exact[packet_type]:
        raise E6V3ProtocolError(
            f"invalid packet payload size for type {packet_type}: {payload_size}"
        )
    if packet_type == E6V3_RGB_CODEC_CONFIG and not (
        0 < payload_size <= E6V3_MAX_RGB_CONFIG_SIZE
    ):
        raise E6V3ProtocolError(f"invalid RGB codec config size: {payload_size}")
    if packet_type == E6V3_RGB_FRAME and payload_size <= E6V3_RGB_METADATA_SIZE:
        raise E6V3ProtocolError("RGB frame has no H.265 access unit")
    if packet_type == E6V3_AUDIO_CODEC_CONFIG and not (
        0 < payload_size <= E6V3_MAX_AUDIO_CONFIG_SIZE
    ):
        raise E6V3ProtocolError(f"invalid audio codec config size: {payload_size}")
    if packet_type == E6V3_AUDIO_FRAME and payload_size <= E6V3_AUDIO_METADATA_SIZE:
        raise E6V3ProtocolError("audio frame has no AAC access unit")

    return E6V3Header(
        packet_type=packet_type,
        flags=flags,
        payload_size=payload_size,
        session_id=session_id,
        sequence_number=sequence_number,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
    )


def parse_e6v3_session_start(
    header: E6V3Header, payload: bytes
) -> E6V3Session:
    """Validate the fixed 3200x1200@60 H.265 + required AAC session."""

    if header.packet_type != E6V3_SESSION_START_TYPE:
        raise E6V3ProtocolError("payload is not SESSION_START")
    if len(payload) != E6V3_SESSION_START.size:
        raise E6V3ProtocolError("invalid Protocol V3 SESSION_START payload")
    values = E6V3_SESSION_START.unpack(payload)
    (
        width,
        height,
        fps_numerator,
        fps_denominator,
        rgb_codec,
        clock_domain,
        rgb_metadata_size,
        stream_flags,
        audio_codec,
        audio_sample_rate,
        audio_channels,
        audio_bit_rate,
        anchor_boot_ns,
        anchor_realtime_ns,
        anchor_uncertainty_ns,
    ) = values
    expected_media = (
        3200,
        1200,
        60,
        1,
        E6V3_RGB_CODEC_FOURCC,
        E6V3_CLOCK_BOOTTIME,
        E6V3_RGB_METADATA_SIZE,
        E6V3_AUDIO_REQUIRED,
        E6V3_AUDIO_CODEC_FOURCC,
        44100,
        1,
        96000,
    )
    if values[:12] != expected_media:
        raise E6V3ProtocolError(
            "unexpected E6 V3 media contract: "
            f"got={values[:12]!r} expected={expected_media!r}"
        )
    if not anchor_boot_ns or not anchor_realtime_ns or not anchor_uncertainty_ns:
        raise E6V3ProtocolError("invalid Protocol V3 clock anchor")
    if header.timestamp_ns != anchor_boot_ns:
        raise E6V3ProtocolError(
            "SESSION_START header/payload BOOTTIME anchor mismatch"
        )
    return E6V3Session(
        width=width,
        height=height,
        fps_numerator=fps_numerator,
        fps_denominator=fps_denominator,
        clock_anchor_boottime_ns=anchor_boot_ns,
        clock_anchor_realtime_ns=anchor_realtime_ns,
        clock_anchor_uncertainty_ns=anchor_uncertainty_ns,
    )


def parse_e6v3_rgb_frame(
    header: E6V3Header, payload: bytes
) -> E6V3RgbFrame:
    """Split the 96-byte RGB metadata from its complete H.265 access unit."""

    if header.packet_type != E6V3_RGB_FRAME:
        raise E6V3ProtocolError("payload is not RGB_FRAME")
    if len(payload) != header.payload_size or len(payload) <= E6V3_RGB_METADATA_SIZE:
        raise E6V3ProtocolError("invalid RGB_FRAME payload size")
    values = E6V3_RGB_METADATA.unpack_from(payload)
    exposure_start_ns = values[0]
    exposure_duration_ns = values[1]
    controller_sample_ns = values[3]
    controller_residual_ns = values[4]
    exposure_midpoint_ns = exposure_start_ns + exposure_duration_ns // 2
    if not exposure_start_ns or not exposure_duration_ns:
        raise E6V3ProtocolError("RGB exposure timing is unavailable")
    if exposure_midpoint_ns != header.timestamp_ns:
        raise E6V3ProtocolError("RGB timestamp is not the exposure midpoint")
    sample_available = controller_sample_ns != E6V3_UNAVAILABLE_TIMESTAMP_NS
    residual_available = controller_residual_ns != E6V3_UNAVAILABLE_TIMESTAMP_NS
    if sample_available != residual_available:
        raise E6V3ProtocolError("controller time availability differs")
    if sample_available and abs(controller_sample_ns - header.timestamp_ns) != controller_residual_ns:
        raise E6V3ProtocolError("controller residual does not match RGB timestamp")
    pose_expected = sample_available and controller_residual_ns <= 20_000_000
    if pose_expected != bool(header.flags & E6V3_FLAG_POSE_MATCHED):
        raise E6V3ProtocolError("POSE_MATCHED flag does not match RGB metadata")
    access_unit = bytes(payload[E6V3_RGB_METADATA_SIZE:])
    if not access_unit:
        raise E6V3ProtocolError("RGB frame has no H.265 access unit")
    return E6V3RgbFrame(
        exposure_start_boottime_ns=exposure_start_ns,
        exposure_duration_ns=exposure_duration_ns,
        gain=values[2],
        controller_sample_boottime_ns=controller_sample_ns,
        controller_residual_ns=controller_residual_ns,
        access_unit=access_unit,
    )


@dataclass
class E6V3StreamState:
    """Sequence and ordering checks scoped to one TCP session."""

    session: E6V3Session | None = None
    session_id: int = 0
    next_sequence: int = 0
    rgb_config_seen: bool = False
    audio_config_seen: bool = False
    first_rgb_seen: bool = False

    def validate_header(self, header: E6V3Header) -> None:
        if self.session is None:
            if (
                header.packet_type != E6V3_SESSION_START_TYPE
                or header.sequence_number != 0
                or header.session_id == 0
                or header.frame_id != E6V3_NO_FRAME_ID
                or header.timestamp_ns == 0
            ):
                raise E6V3ProtocolError(
                    "stream must begin with a valid SESSION_START"
                )
            return
        if (
            header.session_id != self.session_id
            or header.sequence_number != self.next_sequence
            or header.sequence_number == 0xFFFFFFFFFFFFFFFF
        ):
            raise E6V3ProtocolError(
                "session id or sequence number changed unexpectedly"
            )
        if header.packet_type == E6V3_SESSION_START_TYPE:
            raise E6V3ProtocolError("duplicate SESSION_START")

        if header.packet_type in (
            E6V3_RGB_CODEC_CONFIG,
            E6V3_AUDIO_CODEC_CONFIG,
        ):
            if header.frame_id != E6V3_NO_FRAME_ID or header.timestamp_ns != 0:
                raise E6V3ProtocolError("codec config has frame id or timestamp")
        elif header.packet_type == E6V3_RGB_FRAME:
            if header.frame_id == E6V3_NO_FRAME_ID or header.timestamp_ns == 0:
                raise E6V3ProtocolError("RGB frame has invalid id or timestamp")
            if not self.rgb_config_seen:
                raise E6V3ProtocolError("RGB frame arrived before codec config")
            if not self.first_rgb_seen and not (
                header.flags & E6V3_FLAG_KEY_FRAME
            ):
                raise E6V3ProtocolError("first RGB frame is not a keyframe")
        elif header.packet_type == E6V3_AUDIO_FRAME:
            if header.frame_id != E6V3_NO_FRAME_ID or header.timestamp_ns == 0:
                raise E6V3ProtocolError("audio frame has invalid id or timestamp")
            if not self.audio_config_seen:
                raise E6V3ProtocolError("audio frame arrived before codec config")
        elif header.packet_type == E6V3_SESSION_STOP:
            if header.frame_id != E6V3_NO_FRAME_ID or header.timestamp_ns == 0:
                raise E6V3ProtocolError("SESSION_STOP has invalid id or timestamp")

    def accept_start(self, header: E6V3Header, payload: bytes) -> None:
        self.validate_header(header)
        self.session = parse_e6v3_session_start(header, payload)
        self.session_id = header.session_id
        self.next_sequence = 1

    def commit(self, header: E6V3Header) -> None:
        if header.packet_type == E6V3_RGB_CODEC_CONFIG:
            self.rgb_config_seen = True
        elif header.packet_type == E6V3_AUDIO_CODEC_CONFIG:
            self.audio_config_seen = True
        elif header.packet_type == E6V3_RGB_FRAME:
            self.first_rgb_seen = True
        self.next_sequence += 1


def read_e6v3_exact(
    connection: socket.socket,
    size: int,
    *,
    stop: threading.Event | None = None,
) -> bytes:
    """Read exactly one framed field; partial packet timeout is fatal."""

    if size < 0:
        raise ValueError("read size must not be negative")
    data = bytearray()
    while len(data) < size:
        if stop is not None and stop.is_set():
            raise InterruptedError("E6 reader is stopping")
        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout:
            if data:
                raise TimeoutError(
                    f"E6 packet stalled after {len(data)}/{size} bytes"
                )
            continue
        if not chunk:
            raise EOFError(f"E6 stream ended after {len(data)}/{size} bytes")
        data.extend(chunk)
    return bytes(data)


@dataclass(frozen=True)
class E6HistoryFrame:
    image: np.ndarray
    sequence: int
    decode_wall_time_ns: int
    decode_monotonic_ns: int
    alignment_wall_time_ns: int
    input_shape: tuple[int, ...]
    selected_eye_shape: tuple[int, ...]
    layout: str
    protocol_frame_id: int | None = None
    source_boottime_ns: int | None = None
    source_e6_realtime_ns: int | None = None
    clock_anchor_uncertainty_ns: int | None = None


class RetryableCameraObservation(RuntimeError):
    """A transient pre-model camera condition that requires a fresh XR sample."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.diagnostics = dict(diagnostics or {})


class E6TcpCamHigh:
    """Reconnect-safe latest-frame source for the E6 right RGB camera."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18554,
        *,
        eye: str = "right",
        reconnect_seconds: float = 1.0,
        socket_timeout: float = 2.0,
        history_seconds: float = 8.0,
        timestamp_offset_ms: float | None = None,
    ) -> None:
        if eye not in ("left", "right"):
            raise ValueError("E6 eye must be 'left' or 'right'")
        self.host = host
        self.port = port
        self.eye = eye
        if not np.isfinite(history_seconds) or history_seconds <= 0.0:
            raise ValueError("E6 history_seconds must be positive")
        if timestamp_offset_ms is not None and not np.isfinite(
            timestamp_offset_ms
        ):
            raise ValueError("E6 timestamp_offset_ms must be finite")
        self._reconnect_seconds = reconnect_seconds
        self._socket_timeout = socket_timeout
        self._history_seconds = float(history_seconds)
        # Protocol V3 maps BOOTTIME to the E6 headset's REALTIME, not to the
        # 5090 clock. Source timestamps are therefore enabled only when an
        # explicit ``5090 realtime - E6 realtime`` calibration is supplied.
        self._timestamp_offset_ns = (
            None
            if timestamp_offset_ms is None
            else int(round(timestamp_offset_ms * 1e6))
        )
        self._lock = threading.Lock()
        self._history: deque[E6HistoryFrame] = deque()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._frame_wall_time_ns = 0
        self._frame_monotonic_ns = 0
        self._frame_sequence = 0
        self._protocol_frame_id: int | None = None
        self._source_boottime_ns: int | None = None
        self._source_e6_realtime_ns: int | None = None
        self._clock_anchor_uncertainty_ns: int | None = None
        self._input_shape: tuple[int, ...] | None = None
        self._selected_eye_shape: tuple[int, ...] | None = None
        self._layout = "unknown"
        self._error = "stream has not produced a frame"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="e6-tcp-hevc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            age_ms = None if self._frame is None else (time.monotonic() - self._frame_time) * 1000.0
            return {
                "endpoint": self.endpoint,
                "transport": E6V3_TRANSPORT,
                "eye": self.eye,
                "ready": self._frame is not None,
                "frame_sequence": self._frame_sequence,
                "protocol_frame_id": self._protocol_frame_id,
                "frame_wall_time_ns": self._frame_wall_time_ns or None,
                "frame_monotonic_ns": self._frame_monotonic_ns or None,
                "source_boottime_ns": self._source_boottime_ns,
                "source_e6_realtime_ns": self._source_e6_realtime_ns,
                "clock_anchor_uncertainty_ns": (
                    self._clock_anchor_uncertainty_ns
                ),
                "age_ms": age_ms,
                "input_shape": None if self._input_shape is None else list(self._input_shape),
                "selected_eye_shape": (
                    None if self._selected_eye_shape is None else list(self._selected_eye_shape)
                ),
                "output_shape": None if self._frame is None else list(self._frame.shape),
                "layout": self._layout,
                "history_frames": len(self._history),
                "history_seconds": self._history_seconds,
                "history_oldest_alignment_wall_time_ns": (
                    None if not self._history else self._history[0].alignment_wall_time_ns
                ),
                "history_newest_alignment_wall_time_ns": (
                    None if not self._history else self._history[-1].alignment_wall_time_ns
                ),
                "timestamp_offset_ms": (
                    None
                    if self._timestamp_offset_ns is None
                    else self._timestamp_offset_ns / 1e6
                ),
                "timestamp_semantics": (
                    E6V3_DECODE_TIMESTAMP_SEMANTICS
                    if self._timestamp_offset_ns is None
                    else E6V3_TIMESTAMP_SEMANTICS
                ),
                "error": self._error,
            }

    def latest(self, *, max_age_ms: float) -> np.ndarray:
        frame, _ = self.latest_snapshot(max_age_ms=max_age_ms)
        return frame

    def latest_snapshot(self, *, max_age_ms: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one frame and its metadata under the same lock.

        This prevents the recorder from associating an injected image with the
        age or sequence number of a newer E6 frame.
        """
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"E6 cam_h unavailable: {self._error}")
            now_monotonic_ns = time.monotonic_ns()
            age_ms = (now_monotonic_ns - self._frame_monotonic_ns) / 1e6
            if age_ms > max_age_ms:
                raise RuntimeError(
                    f"E6 cam_h stale: age={age_ms:.1f}ms limit={max_age_ms:.1f}ms; {self._error}"
                )
            return self._frame.copy(), {
                "frame_sequence": self._frame_sequence,
                "protocol_frame_id": self._protocol_frame_id,
                "frame_wall_time_ns": self._frame_wall_time_ns,
                "frame_monotonic_ns": self._frame_monotonic_ns,
                "source_boottime_ns": self._source_boottime_ns,
                "source_e6_realtime_ns": self._source_e6_realtime_ns,
                "clock_anchor_uncertainty_ns": (
                    self._clock_anchor_uncertainty_ns
                ),
                "alignment_wall_time_ns": self._history[-1].alignment_wall_time_ns,
                "age_ms": age_ms,
                "input_shape": (
                    None if self._input_shape is None else list(self._input_shape)
                ),
                "selected_eye_shape": (
                    None
                    if self._selected_eye_shape is None
                    else list(self._selected_eye_shape)
                ),
                "output_shape": list(self._frame.shape),
                "layout": self._layout,
                "eye": self.eye,
                "endpoint": self.endpoint,
                "timestamp_offset_ms": (
                    None
                    if self._timestamp_offset_ns is None
                    else self._timestamp_offset_ns / 1e6
                ),
                "timestamp_semantics": (
                    E6V3_DECODE_TIMESTAMP_SEMANTICS
                    if self._timestamp_offset_ns is None
                    else E6V3_TIMESTAMP_SEMANTICS
                ),
            }

    def aligned_snapshot(
        self,
        *,
        reference_wall_times_ns: dict[str, int],
        max_pairwise_skew_ms: float,
        max_latest_age_ms: float,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Select a historical E6 frame aligned to the wrist timestamps.

        The latest frame is checked separately for stream liveness.  A request
        may legitimately select a several-second-old historical frame after a
        slow wrist upload, but a dead E6 stream is never accepted.
        """

        if not reference_wall_times_ns:
            raise ValueError("camera sync request has no wrist timestamps")
        if not np.isfinite(max_pairwise_skew_ms) or max_pairwise_skew_ms <= 0.0:
            raise ValueError("max_pairwise_skew_ms must be positive")
        with self._lock:
            if not self._history:
                raise RetryableCameraObservation(
                    "e6_unavailable",
                    f"E6 cam_h unavailable: {self._error}",
                )
            latest = self._history[-1]
            now_monotonic_ns = time.monotonic_ns()
            latest_age_ms = (now_monotonic_ns - latest.decode_monotonic_ns) / 1e6
            if latest_age_ms > max_latest_age_ms:
                raise RetryableCameraObservation(
                    "e6_stream_stale",
                    (
                        "E6 cam_h stream is stale: "
                        f"latest_age={latest_age_ms:.1f}ms "
                        f"limit={max_latest_age_ms:.1f}ms; {self._error}"
                    ),
                    diagnostics={
                        "latest_stream_age_ms": latest_age_ms,
                        "max_latest_age_ms": max_latest_age_ms,
                    },
                )
            history = tuple(self._history)
            candidate_times = [frame.alignment_wall_time_ns for frame in history]
            references = [int(value) for value in reference_wall_times_ns.values()]
            index, signed_midpoint_ns, span_ns = select_best_timestamp(
                candidate_times, references
            )
            selected = history[index]
            limit_ns = int(round(max_pairwise_skew_ms * 1e6))
            if span_ns > limit_ns:
                raise RetryableCameraObservation(
                    "three_camera_not_aligned",
                    (
                        "three-camera timestamp alignment failed: "
                        f"best_span_ms={span_ns / 1e6:.3f} "
                        f"limit_ms={max_pairwise_skew_ms:.3f} "
                        f"selected_ns={selected.alignment_wall_time_ns} "
                        f"signed_to_wrist_midpoint_ms={signed_midpoint_ns / 1e6:+.3f} "
                        f"target_ns={references} history="
                        f"[{history[0].alignment_wall_time_ns},"
                        f"{history[-1].alignment_wall_time_ns}]"
                    ),
                    diagnostics={
                        "best_span_ms": span_ns / 1e6,
                        "max_pairwise_skew_ms": max_pairwise_skew_ms,
                        "selected_wall_time_ns": selected.alignment_wall_time_ns,
                        "signed_to_wrist_midpoint_ms": signed_midpoint_ns / 1e6,
                    },
                )
            selected_copy = selected.image.copy()

        metadata = {
            "frame_sequence": selected.sequence,
            "protocol_frame_id": selected.protocol_frame_id,
            "frame_wall_time_ns": selected.decode_wall_time_ns,
            "frame_monotonic_ns": selected.decode_monotonic_ns,
            "source_boottime_ns": selected.source_boottime_ns,
            "source_e6_realtime_ns": selected.source_e6_realtime_ns,
            "clock_anchor_uncertainty_ns": selected.clock_anchor_uncertainty_ns,
            "alignment_wall_time_ns": selected.alignment_wall_time_ns,
            "age_at_selection_ms": (
                time.monotonic_ns() - selected.decode_monotonic_ns
            )
            / 1e6,
            "latest_stream_age_ms": latest_age_ms,
            "signed_to_wrist_midpoint_ms": signed_midpoint_ns / 1e6,
            "input_shape": list(selected.input_shape),
            "selected_eye_shape": list(selected.selected_eye_shape),
            "output_shape": list(selected.image.shape),
            "layout": selected.layout,
            "eye": self.eye,
            "endpoint": self.endpoint,
            "timestamp_offset_ms": (
                None
                if self._timestamp_offset_ns is None
                else self._timestamp_offset_ns / 1e6
            ),
            "timestamp_semantics": (
                E6V3_DECODE_TIMESTAMP_SEMANTICS
                if self._timestamp_offset_ns is None
                else E6V3_TIMESTAMP_SEMANTICS
            ),
        }
        alignment = {
            "max_pairwise_skew_ns": int(span_ns),
            "max_pairwise_skew_ms": span_ns / 1e6,
        }
        return selected_copy, metadata, alignment

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _select_cam_h(self, decoded_rgb: np.ndarray) -> tuple[np.ndarray, str]:
        """Accept the new single-eye stream; retain stereo compatibility.

        The new RGB app normally emits one already-selected eye.  If a future
        build returns the legacy side-by-side layout, select the configured eye
        without changing the policy contract.
        """
        image = np.asarray(decoded_rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"decoded E6 image must be HxWx3 uint8, got {image.shape} {image.dtype}")
        height, width = image.shape[:2]
        if width == 2 * height * 4 // 3:
            midpoint = width // 2
            selected = image[:, :midpoint] if self.eye == "left" else image[:, midpoint:]
            layout = "stereo_side_by_side"
        else:
            selected = image
            layout = "single_eye"
        # taskumi2 training first cropped the 3200x1200 stereo video to the
        # 1600x1200 right eye, then stored the model-facing video at 640x480.
        # Preserve that exact intermediate input size before OpenPI performs
        # its own model resize.
        resized = cv2.resize(selected, TRAINING_CAM_H_SIZE, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized), layout

    def _set_frame(
        self,
        decoded_rgb: np.ndarray,
        *,
        protocol_frame_id: int,
        source_boottime_ns: int,
        source_e6_realtime_ns: int,
        clock_anchor_uncertainty_ns: int,
    ) -> None:
        frame, layout = self._select_cam_h(decoded_rgb)
        frame_wall_time_ns = time.time_ns()
        frame_monotonic_ns = time.monotonic_ns()
        alignment_wall_time_ns = (
            frame_wall_time_ns
            if self._timestamp_offset_ns is None
            else source_e6_realtime_ns + self._timestamp_offset_ns
        )
        with self._lock:
            first = self._frame is None
            next_sequence = self._frame_sequence + 1
            selected_eye_shape = (
                decoded_rgb.shape[0],
                decoded_rgb.shape[1] // 2 if layout == "stereo_side_by_side" else decoded_rgb.shape[1],
                decoded_rgb.shape[2],
            )
            history_frame = E6HistoryFrame(
                image=frame,
                sequence=next_sequence,
                decode_wall_time_ns=frame_wall_time_ns,
                decode_monotonic_ns=frame_monotonic_ns,
                alignment_wall_time_ns=alignment_wall_time_ns,
                input_shape=tuple(decoded_rgb.shape),
                selected_eye_shape=selected_eye_shape,
                layout=layout,
                protocol_frame_id=protocol_frame_id,
                source_boottime_ns=source_boottime_ns,
                source_e6_realtime_ns=source_e6_realtime_ns,
                clock_anchor_uncertainty_ns=clock_anchor_uncertainty_ns,
            )
            self._history.append(history_frame)
            cutoff_ns = frame_monotonic_ns - int(self._history_seconds * 1e9)
            while (
                len(self._history) > 1
                and self._history[0].decode_monotonic_ns < cutoff_ns
            ):
                self._history.popleft()
            self._frame = frame
            self._frame_time = frame_monotonic_ns / 1e9
            self._frame_wall_time_ns = frame_wall_time_ns
            self._frame_monotonic_ns = frame_monotonic_ns
            self._frame_sequence = next_sequence
            self._protocol_frame_id = protocol_frame_id
            self._source_boottime_ns = source_boottime_ns
            self._source_e6_realtime_ns = source_e6_realtime_ns
            self._clock_anchor_uncertainty_ns = clock_anchor_uncertainty_ns
            self._input_shape = tuple(decoded_rgb.shape)
            self._selected_eye_shape = selected_eye_shape
            self._layout = layout
            self._error = ""
        if first:
            LOGGER.info(
                "E6 cam_h ready: endpoint=%s eye=%s layout=%s input=%s output=%s",
                self.endpoint,
                self.eye,
                layout,
                decoded_rgb.shape,
                frame.shape,
            )

    def _consume_v3_session(self, connection: socket.socket) -> None:
        """Consume one validated V3 session until STOP or disconnect."""

        if av is None:
            raise RuntimeError("PyAV is required to decode E6 H.265 frames")
        state = E6V3StreamState()
        codec: Any | None = None
        # Decoder PTS is the source frame id, allowing future reordered output
        # to retain the exact exposure timestamp that belongs to that frame.
        pending_timing: dict[int, tuple[int, int, int]] = {}
        while not self._stop.is_set():
            header = parse_e6v3_header(
                read_e6v3_exact(
                    connection, E6V3_HEADER.size, stop=self._stop
                )
            )
            payload = read_e6v3_exact(
                connection, header.payload_size, stop=self._stop
            )
            if state.session is None:
                state.accept_start(header, payload)
                LOGGER.info(
                    "E6 V3 session ready: session_id=%d rgb=%dx%d@%g "
                    "clock_anchor_uncertainty_ms=%.6f",
                    state.session_id,
                    state.session.width,
                    state.session.height,
                    state.session.fps_numerator
                    / state.session.fps_denominator,
                    state.session.clock_anchor_uncertainty_ns / 1e6,
                )
                continue

            state.validate_header(header)
            if header.packet_type == E6V3_RGB_CODEC_CONFIG:
                if state.rgb_config_seen:
                    raise E6V3ProtocolError("duplicate RGB codec config")
                codec = av.CodecContext.create("hevc", "r")
                # The V3 config is Annex-B VPS/SPS/PPS. Supplying it as
                # extradata and decoding complete framed access units avoids
                # exposing any E6 header/metadata bytes to FFmpeg.
                codec.extradata = payload
                pending_timing.clear()
            elif header.packet_type == E6V3_RGB_FRAME:
                if codec is None or state.session is None:
                    raise E6V3ProtocolError("RGB decoder is not configured")
                rgb = parse_e6v3_rgb_frame(header, payload)
                source_realtime_ns = state.session.source_realtime_ns(
                    header.timestamp_ns
                )
                pending_timing[header.frame_id] = (
                    header.timestamp_ns,
                    source_realtime_ns,
                    state.session.clock_anchor_uncertainty_ns,
                )
                packet = av.Packet(rgb.access_unit)
                packet.pts = header.frame_id
                packet.dts = header.frame_id
                decoded_frames = codec.decode(packet)
                for decoded in decoded_frames:
                    if decoded.pts is None:
                        if len(decoded_frames) != 1:
                            raise E6V3ProtocolError(
                                "decoder returned multiple frames without PTS"
                            )
                        decoded_frame_id = header.frame_id
                    else:
                        decoded_frame_id = int(decoded.pts)
                    timing = pending_timing.pop(decoded_frame_id, None)
                    if timing is None:
                        raise E6V3ProtocolError(
                            f"decoder returned unknown frame id {decoded_frame_id}"
                        )
                    source_boot_ns, source_wall_ns, uncertainty_ns = timing
                    self._set_frame(
                        decoded.to_ndarray(format="rgb24"),
                        protocol_frame_id=decoded_frame_id,
                        source_boottime_ns=source_boot_ns,
                        source_e6_realtime_ns=source_wall_ns,
                        clock_anchor_uncertainty_ns=uncertainty_ns,
                    )
                if len(pending_timing) > 32:
                    raise E6V3ProtocolError(
                        "H.265 decoder buffered more than 32 timestamped frames"
                    )
            elif header.packet_type == E6V3_AUDIO_CODEC_CONFIG:
                if state.audio_config_seen:
                    raise E6V3ProtocolError("duplicate audio codec config")
            elif header.packet_type == E6V3_SESSION_STOP:
                LOGGER.info(
                    "E6 V3 session stopped: session_id=%d", state.session_id
                )
                return
            # AUDIO_FRAME is intentionally ignored after framing/order checks.
            state.commit(header)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                LOGGER.info("Connecting E6 cam_h: %s", self.endpoint)
                with socket.create_connection(
                    (self.host, self.port), timeout=self._socket_timeout
                ) as connection:
                    connection.settimeout(self._socket_timeout)
                    self._consume_v3_session(connection)
            except InterruptedError:
                break
            except Exception as exc:  # transport/decode boundary; reconnect safely
                message = f"{type(exc).__name__}: {exc}"
                self._set_error(message)
                LOGGER.warning("E6 cam_h disconnected (%s); retrying", message)
                self._stop.wait(self._reconnect_seconds)


class HeadCameraInjectingPolicy:
    """Make the 5090-owned live E6 frame authoritative for cam_h."""

    def __init__(
        self,
        policy: Any,
        source: E6TcpCamHigh,
        *,
        max_age_ms: float,
        max_sync_skew_ms: float,
        max_observation_transport_age_ms: float,
        recorder: InferenceRecorder | None = None,
        fixed_prompt: str | None = None,
        cam_high_roi: CamHighRoiConfig | None = None,
        expected_action_shape: tuple[int, int] = (50, 30),
        expected_observation_interval_ms: float = 100.0,
        observation_interval_tolerance_ms: float = 45.0,
    ) -> None:
        if (
            not isinstance(expected_action_shape, tuple)
            or len(expected_action_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in expected_action_shape
            )
        ):
            raise ValueError(
                "expected_action_shape must contain two positive integers: "
                f"{expected_action_shape!r}"
            )
        try:
            expected_interval_ms = float(expected_observation_interval_ms)
            interval_tolerance_ms = float(observation_interval_tolerance_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "policy observation interval values must be numeric"
            ) from exc
        if (
            isinstance(expected_observation_interval_ms, bool)
            or isinstance(observation_interval_tolerance_ms, bool)
            or not np.isfinite(expected_interval_ms)
            or expected_interval_ms <= 0.0
            or not np.isfinite(interval_tolerance_ms)
            or interval_tolerance_ms < 0.0
            or interval_tolerance_ms >= expected_interval_ms
        ):
            raise ValueError(
                "invalid policy observation interval contract: "
                f"expected={expected_observation_interval_ms!r}ms "
                f"tolerance={observation_interval_tolerance_ms!r}ms"
            )
        self._policy = policy
        self._source = source
        self._max_age_ms = max_age_ms
        self._max_sync_skew_ms = max_sync_skew_ms
        # Retain the constructor argument so an older launcher fails neither at
        # import nor startup. Positive transport age is diagnostic-only now:
        # the robot waits at a chunk boundary while this request is in flight.
        del max_observation_transport_age_ms
        self._recorder = recorder
        self._fixed_prompt = fixed_prompt
        self._cam_high_roi = cam_high_roi
        self._expected_action_shape = expected_action_shape
        if (
            recorder is not None
            and recorder.action_shape != self._expected_action_shape
        ):
            raise ValueError(
                "policy and recorder action shapes differ: "
                f"policy={self._expected_action_shape} "
                f"recorder={recorder.action_shape}"
            )
        self._expected_observation_interval_ms = float(
            expected_interval_ms
        )
        self._observation_interval_tolerance_ms = float(
            interval_tolerance_ms
        )
        status = source.status()
        self.metadata = dict(getattr(policy, "metadata", {}) or {})
        self.metadata.update(
            {
                "cam_high_source": "e6_right_rgb_on_policy_host",
                "e6_eye": source.eye,
                "e6_endpoint": source.endpoint,
                "e6_transport": E6V3_TRANSPORT,
                "e6_layout": status["layout"],
                "e6_input_shape": status["input_shape"],
                "e6_output_shape": status["output_shape"],
                "e6_frame_live": True,
                "e6_max_age_ms": max_age_ms,
                "camera_sync_required": True,
                "camera_sync_schema": SYNC_SCHEMA,
                "camera_sync_clock_domain": CLOCK_DOMAIN,
                "camera_clock_sync_required": True,
                "camera_clock_sync_schema": CLOCK_SYNC_SCHEMA,
                "camera_clock_sync_max_uncertainty_ms": (
                    CLOCK_SYNC_MAX_UNCERTAINTY_MS
                ),
                "camera_clock_sync_age_policy": OBSERVATION_TRANSPORT_AGE_POLICY,
                "camera_sync_max_skew_ms": max_sync_skew_ms,
                "camera_sync_observation_transport_age_policy": (
                    OBSERVATION_TRANSPORT_AGE_POLICY
                ),
                "observation_retry_schema": OBSERVATION_RETRY_SCHEMA,
                "state_sync_max_skew_ms": DEFAULT_STATE_SYNC_MAX_SKEW_MS,
                "state_observation_interval_ms": (
                    self._expected_observation_interval_ms
                ),
                "state_observation_interval_tolerance_ms": (
                    self._observation_interval_tolerance_ms
                ),
                "live_actuation_allowed": True,
                "robot_image_keys": ["cam_left_wrist", "cam_right_wrist"],
                "wrist_image_transport_schema": WRIST_TRANSPORT_SCHEMA,
                "wrist_image_transport": "original_ros_compressed_bytes",
                "transport_probe_schema": TRANSPORT_PROBE_SCHEMA,
                "prompt_source": (
                    "checkpoint_profile" if fixed_prompt is not None else "client"
                ),
                "cam_high_model_preprocess": (
                    "normalized_roi_before_resize_v1"
                    if cam_high_roi is not None
                    else "full_frame_v1"
                ),
            }
        )
        if recorder is not None:
            self.metadata.update(recorder.metadata)

    @staticmethod
    def _retry_response(
        reason: str,
        message: str,
        *,
        server_receive_wall_time_ns: int,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retry = {
            "schema": OBSERVATION_RETRY_SCHEMA,
            "reason": reason,
            "message": message,
            "server_receive_wall_time_ns": server_receive_wall_time_ns,
            "retry_after_ms": 10.0,
        }
        retry.update(diagnostics or {})
        LOGGER.warning("Observation retry requested: reason=%s %s", reason, message)
        return {"observation_retry": retry}

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        server_handler_receive_wall_time_ns = time.time_ns()
        probe_raw = observation.get("_transport_probe")
        if probe_raw is not None:
            if set(observation) != {"_transport_probe"}:
                raise ValueError("transport probe must not contain policy inputs")
            if not isinstance(probe_raw, dict):
                raise ValueError("transport probe must be a mapping")
            if probe_raw.get("schema") != TRANSPORT_PROBE_SCHEMA:
                raise ValueError(
                    f"unexpected transport probe schema: {probe_raw.get('schema')!r}"
                )
            sequence = probe_raw.get("sequence")
            padding = probe_raw.get("padding")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence <= 0
            ):
                raise ValueError(f"invalid transport probe sequence: {sequence!r}")
            if not isinstance(padding, bytes) or not (
                1 <= len(padding) <= MAX_TRANSPORT_PROBE_BYTES
            ):
                raise ValueError("invalid transport probe padding")
            server_handler_send_wall_time_ns = time.time_ns()
            return {
                "transport_probe": {
                    "schema": TRANSPORT_PROBE_SCHEMA,
                    "sequence": sequence,
                    "payload_bytes": len(padding),
                    # Keep the old field until every XR client has migrated.
                    "server_wall_time_ns": server_handler_send_wall_time_ns,
                    "server_receive_wall_time_ns": (
                        server_handler_receive_wall_time_ns
                    ),
                    "server_send_wall_time_ns": server_handler_send_wall_time_ns,
                }
            }
        client_recording_raw = observation.get("_recording", {})
        if client_recording_raw is not None and not isinstance(
            client_recording_raw, dict
        ):
            raise ValueError("_recording must be a mapping when provided")
        try:
            client_sync = parse_client_sync(
                observation.get("_sync"),
                server_receive_wall_time_ns=(
                    server_handler_receive_wall_time_ns
                ),
                max_clock_uncertainty_ms=CLOCK_SYNC_MAX_UNCERTAINTY_MS,
                max_sync_skew_ms=self._max_sync_skew_ms,
                state_sync_max_skew_ms=DEFAULT_STATE_SYNC_MAX_SKEW_MS,
                expected_observation_interval_ms=(
                    self._expected_observation_interval_ms
                ),
                observation_interval_tolerance_ms=(
                    self._observation_interval_tolerance_ms
                ),
            )
        except ClockSyncUncertainForAlignment as exc:
            return self._retry_response(
                "clock_sync_uncertain",
                str(exc),
                server_receive_wall_time_ns=server_handler_receive_wall_time_ns,
                diagnostics={
                    "clock_uncertainty_ms": exc.uncertainty_ms,
                    "max_clock_uncertainty_ms": exc.max_uncertainty_ms,
                },
            )
        images = dict(observation.get("images", {}))
        for key in ("cam_left_wrist", "cam_right_wrist"):
            if key not in images:
                raise ValueError(f"observation images are missing {key}")
            images[key] = decode_wrist_transport(images[key])
        try:
            cam_high, e6_snapshot, alignment = self._source.aligned_snapshot(
                reference_wall_times_ns=client_sync.reference_wall_times_ns,
                max_pairwise_skew_ms=client_sync.effective_sync_limit_ms,
                max_latest_age_ms=self._max_age_ms,
            )
        except RetryableCameraObservation as exc:
            return self._retry_response(
                exc.reason,
                str(exc),
                server_receive_wall_time_ns=(
                    client_sync.server_receive_wall_time_ns
                ),
                diagnostics={
                    "observation_transport_age_ms": (
                        client_sync.observation_transport_age_ms
                    ),
                    "worst_observation_transport_age_ms": (
                        client_sync.worst_observation_transport_age_ms
                    ),
                    "source_to_send_ms": client_sync.source_to_send_ms,
                    "send_to_receive_ms": client_sync.send_to_receive_ms,
                    **exc.diagnostics,
                },
            )
        camera_sync = build_three_camera_sync_proof(
            client_sync,
            e6_snapshot=e6_snapshot,
            alignment=alignment,
        )
        images["cam_high"] = cam_high
        merged = dict(observation)
        # Private transport metadata must never enter the OpenPI transforms.
        merged.pop("_recording", None)
        merged.pop("_sync", None)
        merged["images"] = images
        if self._fixed_prompt is not None:
            merged["prompt"] = self._fixed_prompt
        captured = None
        if self._recorder is not None:
            model_images: dict[str, np.ndarray] = {}
            source_context: dict[str, Any] = {
                "e6": e6_snapshot,
                "camera_sync": camera_sync,
            }
            if self._cam_high_roi is not None:
                roi_result = crop_cam_high_roi(
                    cam_high, config=self._cam_high_roi
                )
                model_images["cam_high_roi_pre_resize"] = roi_result.image
                source_context["cam_high_preprocess"] = {
                    **roi_result.metadata(),
                    "recorded_model_image": "cam_high_roi_pre_resize.png",
                    "recorded_source_image": "cam_high.png",
                    "openpi_next_transform": "ResizeImages(224,224)",
                }
            captured = self._recorder.capture(
                merged,
                model_images=model_images,
                client_recording=client_recording_raw or {},
                source_context=source_context,
            )
        try:
            result = self._policy.infer(merged)
            if not isinstance(result, Mapping):
                raise RuntimeError(
                    "policy returned a non-mapping response: "
                    f"{type(result).__name__}"
                )
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if (
                actions.shape != self._expected_action_shape
                or not np.all(np.isfinite(actions))
            ):
                raise RuntimeError(
                    "policy returned invalid actions: "
                    f"expected={self._expected_action_shape} actual={actions.shape}"
                )
        except Exception as exc:
            completed_ns = time.monotonic_ns()
            if self._recorder is not None and captured is not None:
                self._recorder.record_failure(
                    captured, exc, infer_completed_monotonic_ns=completed_ns
                )
            raise
        completed_ns = time.monotonic_ns()
        if self._recorder is None or captured is None:
            response = dict(result)
            response["camera_sync"] = camera_sync
            return response
        recording = self._recorder.record_success(
            captured, result, infer_completed_monotonic_ns=completed_ns
        )
        response = dict(result)
        response["recording"] = recording
        response["camera_sync"] = camera_sync
        return response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one decoded right-eye frame from E6 Stream Protocol V3"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18554)
    parser.add_argument("--eye", choices=("left", "right"), default="right")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = E6TcpCamHigh(args.host, args.port, eye=args.eye)
    source.start()
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if source.status()["ready"]:
                break
            time.sleep(0.05)
        status = source.status()
        if not status["ready"]:
            raise RuntimeError(f"E6 did not produce a frame: {status}")
        frame = source.latest(max_age_ms=2000.0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write {args.output}")
        print(f"E6_CAM_H_OK output={args.output} status={status}")
    finally:
        source.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
