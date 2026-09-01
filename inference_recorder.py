#!/usr/bin/env python3
"""Lossless, atomic policy-I/O recording for UMI live inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import threading
import time
from typing import Any, Mapping
import uuid

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
RECORD_SCHEMA = "umi_policy_io_record_v2"
IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_SHAPE = (50, 30)
STATE_SHAPE = (30,)

POSE9_SUFFIXES = (
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
)
HAND6_SUFFIXES = (
    "thumb_deg",
    "thumb_aux_deg",
    "index_deg",
    "middle_deg",
    "ring_deg",
    "pinky_deg",
)
ACTION_COLUMNS = (
    *(f"left_{name}" for name in POSE9_SUFFIXES),
    *(f"right_{name}" for name in POSE9_SUFFIXES),
    *(f"left_hand_{name}" for name in HAND6_SUFFIXES),
    *(f"right_hand_{name}" for name in HAND6_SUFFIXES),
)
STATE_COLUMNS = (
    *(f"left_state_{name}" for name in POSE9_SUFFIXES),
    *(f"right_state_{name}" for name in POSE9_SUFFIXES),
    *(f"left_hand_{name}" for name in HAND6_SUFFIXES),
    *(f"right_hand_{name}" for name in HAND6_SUFFIXES),
)


def _utc_iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _policy_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must have 3 axes, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} contains NaN or Inf")
        scale = 255.0 if image.size == 0 or float(np.max(image)) <= 1.0 + 1e-6 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class CapturedInference:
    server_request_index: int
    started_wall_time_ns: int
    started_monotonic_ns: int
    images: dict[str, np.ndarray]
    model_images: dict[str, np.ndarray]
    state30: np.ndarray
    prompt: str
    client_recording: dict[str, Any]
    source_context: dict[str, Any]


class InferenceRecorder:
    """Write one complete inference directory before its response is returned.

    Synchronous commit is intentional. A successful policy response therefore
    always has a fully committed record; a disk error fails closed instead of
    silently dropping a chunk while the robot continues.
    """

    def __init__(
        self,
        root: Path,
        *,
        session_metadata: Mapping[str, Any],
        action_shape: tuple[int, int] = ACTION_SHAPE,
        min_free_gb: float = 20.0,
        png_compression: int = 3,
    ) -> None:
        if not np.isfinite(min_free_gb) or min_free_gb < 0.0:
            raise ValueError("min_free_gb must be finite and non-negative")
        if not 0 <= png_compression <= 9:
            raise ValueError("png_compression must be in [0, 9]")
        if (
            not isinstance(action_shape, tuple)
            or len(action_shape) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in action_shape
            )
        ):
            raise ValueError(
                f"action_shape must contain two positive integers, got {action_shape!r}"
            )
        if action_shape[1] != len(ACTION_COLUMNS):
            raise ValueError(
                "recorder action width does not match its CSV schema: "
                f"shape={action_shape} columns={len(ACTION_COLUMNS)}"
            )
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = int(min_free_gb * (1024**3))
        self.png_compression = int(png_compression)
        self.action_shape = action_shape
        self.action_file_stem = f"actions_{action_shape[0]}x{action_shape[1]}"
        now_ns = time.time_ns()
        stamp = datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        self.session_id = f"{stamp}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.session_dir = self.root / self.session_id
        self.requests_dir = self.session_dir / "requests"
        self.requests_dir.mkdir(parents=True, exist_ok=False)
        self._counter_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._next_index = 1
        self._closed = False
        self._assert_free_space()
        session = {
            "schema": RECORD_SCHEMA,
            "session_id": self.session_id,
            "created_wall_time_ns": now_ns,
            "created_utc": _utc_iso_from_ns(now_ns),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "record_root": str(self.root),
            "image_encoding": "lossless_png_rgb_uint8",
            "min_free_bytes": self.min_free_bytes,
            "action_shape": list(self.action_shape),
            "metadata": _jsonable(dict(session_metadata)),
        }
        self._write_json(self.session_dir / "session.json", session)
        (self.session_dir / "manifest.jsonl").touch(exist_ok=False)
        LOGGER.info("Inference recording enabled: %s", self.session_dir)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "recording_enabled": True,
            "recording_schema": RECORD_SCHEMA,
            "recording_session_id": self.session_id,
            "recording_session_dir": str(self.session_dir),
            "recording_image_encoding": "lossless_png_rgb_uint8",
            "recording_action_shape": list(self.action_shape),
            "recording_action_file_stem": self.action_file_stem,
        }

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("inference recorder is closed")

    def _assert_free_space(self) -> int:
        free = int(shutil.disk_usage(self.root).free)
        if free < self.min_free_bytes:
            raise RuntimeError(
                "inference recorder free space is below the safety floor: "
                f"free={free} required={self.min_free_bytes} root={self.root}"
            )
        return free

    def capture(
        self,
        observation: Mapping[str, Any],
        *,
        model_images: Mapping[str, Any] | None = None,
        client_recording: Mapping[str, Any] | None = None,
        source_context: Mapping[str, Any] | None = None,
    ) -> CapturedInference:
        self._assert_open()
        self._assert_free_space()
        images_raw = observation.get("images")
        if not isinstance(images_raw, Mapping):
            raise ValueError("recorded observation has no image mapping")
        missing = [name for name in IMAGE_KEYS if name not in images_raw]
        if missing:
            raise ValueError(f"recorded observation is missing images: {missing}")
        images = {name: _policy_image(images_raw[name], name).copy() for name in IMAGE_KEYS}
        model_images_captured: dict[str, np.ndarray] = {}
        for name, value in (model_images or {}).items():
            if (
                not isinstance(name, str)
                or not name
                or any(
                    not (character.islower() or character.isdigit() or character == "_")
                    for character in name
                )
                or name in images
            ):
                raise ValueError(f"invalid recorded model image name: {name!r}")
            model_images_captured[name] = _policy_image(value, name).copy()
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != STATE_SHAPE or not np.all(np.isfinite(state)):
            raise ValueError(f"recorded state must be finite {STATE_SHAPE}, got {state.shape}")
        prompt_value = observation.get("prompt", "")
        prompt = (
            prompt_value.decode("utf-8")
            if isinstance(prompt_value, bytes)
            else str(prompt_value)
        )
        with self._counter_lock:
            index = self._next_index
            self._next_index += 1
        return CapturedInference(
            server_request_index=index,
            started_wall_time_ns=time.time_ns(),
            started_monotonic_ns=time.monotonic_ns(),
            images=images,
            model_images=model_images_captured,
            state30=np.ascontiguousarray(state).copy(),
            prompt=prompt,
            client_recording=dict(client_recording or {}),
            source_context=dict(source_context or {}),
        )

    def record_success(
        self,
        captured: CapturedInference,
        result: Mapping[str, Any],
        *,
        infer_completed_monotonic_ns: int,
    ) -> dict[str, Any]:
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != self.action_shape or not np.all(np.isfinite(actions)):
            raise ValueError(
                "recorded actions must be finite "
                f"{self.action_shape}, got {actions.shape}"
            )
        return self._commit(
            captured,
            status="success",
            actions=np.ascontiguousarray(actions).copy(),
            result_metadata={"policy_timing": result.get("policy_timing", {})},
            error=None,
            infer_completed_monotonic_ns=infer_completed_monotonic_ns,
        )

    def record_failure(
        self,
        captured: CapturedInference,
        error: BaseException,
        *,
        infer_completed_monotonic_ns: int,
    ) -> dict[str, Any]:
        return self._commit(
            captured,
            status="inference_error",
            actions=None,
            result_metadata={},
            error={"type": type(error).__name__, "message": str(error)},
            infer_completed_monotonic_ns=infer_completed_monotonic_ns,
        )

    def _commit(
        self,
        captured: CapturedInference,
        *,
        status: str,
        actions: np.ndarray | None,
        result_metadata: Mapping[str, Any],
        error: Mapping[str, Any] | None,
        infer_completed_monotonic_ns: int,
    ) -> dict[str, Any]:
        self._assert_open()
        free_before = self._assert_free_space()
        started_write_ns = time.monotonic_ns()
        name = f"request_{captured.server_request_index:06d}"
        final_dir = self.requests_dir / name
        partial_dir = self.requests_dir / f".{name}.partial-{uuid.uuid4().hex}"
        partial_dir.mkdir(parents=False, exist_ok=False)
        try:
            file_integrity: dict[str, Any] = {}
            for key, image in captured.images.items():
                path = partial_dir / f"{key}.png"
                ok = cv2.imwrite(
                    str(path),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
                )
                if not ok:
                    raise OSError(f"cv2 failed to write {path}")
                file_integrity[key] = {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "raw_sha256": _sha256_array(image),
                    "file": path.name,
                }

            for key, image in captured.model_images.items():
                path = partial_dir / f"{key}.png"
                ok = cv2.imwrite(
                    str(path),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
                )
                if not ok:
                    raise OSError(f"cv2 failed to write {path}")
                file_integrity[key] = {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "raw_sha256": _sha256_array(image),
                    "file": path.name,
                    "role": "model_pre_resize_image",
                }

            np.save(partial_dir / "state30.npy", captured.state30, allow_pickle=False)
            np.savetxt(
                partial_dir / "state30.csv",
                captured.state30[np.newaxis, :],
                delimiter=",",
                header=",".join(STATE_COLUMNS),
                comments="",
                fmt="%.9g",
            )
            file_integrity["state30"] = {
                "shape": list(captured.state30.shape),
                "dtype": str(captured.state30.dtype),
                "raw_sha256": _sha256_array(captured.state30),
                "npy": "state30.npy",
                "csv": "state30.csv",
            }

            if actions is not None:
                actions_npy = f"{self.action_file_stem}.npy"
                actions_csv = f"{self.action_file_stem}.csv"
                np.save(partial_dir / actions_npy, actions, allow_pickle=False)
                action_rows = np.column_stack(
                    [np.arange(self.action_shape[0], dtype=np.int64), actions]
                )
                np.savetxt(
                    partial_dir / actions_csv,
                    action_rows,
                    delimiter=",",
                    header="step," + ",".join(ACTION_COLUMNS),
                    comments="",
                    fmt=["%d", *(["%.9g"] * self.action_shape[1])],
                )
                file_integrity["actions"] = {
                    "shape": list(actions.shape),
                    "dtype": str(actions.dtype),
                    "raw_sha256": _sha256_array(actions),
                    "npy": actions_npy,
                    "csv": actions_csv,
                }

            completed_wall_ns = time.time_ns()
            request_metadata = {
                "schema": RECORD_SCHEMA,
                "session_id": self.session_id,
                "server_request_index": captured.server_request_index,
                "status": status,
                "started_wall_time_ns": captured.started_wall_time_ns,
                "started_utc": _utc_iso_from_ns(captured.started_wall_time_ns),
                "completed_wall_time_ns": completed_wall_ns,
                "completed_utc": _utc_iso_from_ns(completed_wall_ns),
                "started_monotonic_ns": captured.started_monotonic_ns,
                "infer_completed_monotonic_ns": infer_completed_monotonic_ns,
                "policy_wrapper_ms": (
                    infer_completed_monotonic_ns - captured.started_monotonic_ns
                ) / 1e6,
                "prompt": captured.prompt,
                "client_recording": _jsonable(captured.client_recording),
                "source_context": _jsonable(captured.source_context),
                "result_metadata": _jsonable(result_metadata),
                "error": _jsonable(error),
                "free_bytes_before_write": free_before,
                "integrity": file_integrity,
            }
            self._write_json(partial_dir / "request.json", request_metadata)
            os.replace(partial_dir, final_dir)
        except Exception:
            LOGGER.exception("Inference record commit failed; partial retained at %s", partial_dir)
            raise

        write_ms = (time.monotonic_ns() - started_write_ns) / 1e6
        relative_path = str(final_dir.relative_to(self.root))
        response = {
            "schema": RECORD_SCHEMA,
            "session_id": self.session_id,
            "server_request_index": captured.server_request_index,
            "status": status,
            "relative_path": relative_path,
            "write_ms": write_ms,
            "client_run_id": captured.client_recording.get("client_run_id"),
            "client_request_index": captured.client_recording.get(
                "client_request_index"
            ),
            "client_request_role": captured.client_recording.get(
                "client_request_role"
            ),
            "e6_frame_sequence": captured.source_context.get(
                "e6", {}
            ).get("frame_sequence"),
            "e6_frame_age_ms": (
                captured.source_context.get("e6", {}).get("age_at_selection_ms")
                if captured.source_context.get("e6", {}).get(
                    "age_at_selection_ms"
                )
                is not None
                else captured.source_context.get("e6", {}).get("age_ms")
            ),
            "camera_sync_status": captured.source_context.get(
                "camera_sync", {}
            ).get("status"),
            "camera_sync_max_pairwise_skew_ms": captured.source_context.get(
                "camera_sync", {}
            ).get("max_pairwise_skew_ms"),
            "camera_sync_observation_wall_time_ns": captured.source_context.get(
                "camera_sync", {}
            ).get("observation_wall_time_ns"),
        }
        manifest_row = {
            **response,
            "started_wall_time_ns": captured.started_wall_time_ns,
            "started_utc": _utc_iso_from_ns(captured.started_wall_time_ns),
        }
        encoded = json.dumps(_jsonable(manifest_row), ensure_ascii=False, separators=(",", ":"))
        with self._manifest_lock:
            with (self.session_dir / "manifest.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return response

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def close(self) -> None:
        with self._counter_lock:
            if self._closed:
                return
            self._closed = True
            next_index = self._next_index
        now_ns = time.time_ns()
        self._write_json(
            self.session_dir / "session_end.json",
            {
                "schema": RECORD_SCHEMA,
                "session_id": self.session_id,
                "closed_wall_time_ns": now_ns,
                "closed_utc": _utc_iso_from_ns(now_ns),
                "requests_reserved": next_index - 1,
            },
        )
