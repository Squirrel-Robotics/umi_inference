#!/usr/bin/env python3
"""Serve any checkpoint compatible with the current UMI live contract."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import time
import uuid

import numpy as np

from cam_high_roi import CamHighRoiConfig
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

from e6_tcp_cam_h import E6TcpCamHigh, HeadCameraInjectingPolicy
from inference_recorder import InferenceRecorder
import umi_policy_common as common
from umi_live_contract import (
    GENERIC_CHECKPOINT_VALIDATION,
    GENERIC_CONTRACT_SOURCE,
    GENERIC_PROMPT_SOURCE,
    PROMPT_BOX_ONLY,
    checkpoint_params_sha256,
    deployment_contract_metadata,
    checkpoint_params_stat_sha256,
    profile_digest,
    validate_generic_checkpoint_contract,
    validate_policy_contract_metadata,
    validate_requested_cam_high_mode,
    validated_head_camera_preprocess,
)


DEFAULT_RECORD_ROOT = Path("/home/dzq/pi05_inference_records")


def _positive_contract_int(contract: dict, key: str) -> int:
    value = contract.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"deployment contract {key} must be a positive integer: {value!r}")
    return value


def _positive_contract_float(contract: dict, key: str) -> float:
    value = contract.get(key)
    if isinstance(value, bool):
        raise ValueError(f"deployment contract {key} must be finite and positive: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"deployment contract {key} must be finite and positive: {value!r}"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"deployment contract {key} must be finite and positive: {value!r}")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--e6-host", default="127.0.0.1")
    parser.add_argument("--e6-port", type=int, default=18554)
    parser.add_argument("--e6-eye", choices=("left", "right"), default="right")
    parser.add_argument(
        "--cam-high-mode",
        choices=("roi", "full"),
        required=True,
        help="operator-selected cam_high preprocessing; it is not inferred",
    )
    parser.add_argument(
        "--task-prompt",
        default=PROMPT_BOX_ONLY,
        help="task prompt for the generic deployment contract",
    )
    parser.add_argument("--e6-max-age-ms", type=float, default=100.0)
    parser.add_argument("--e6-history-seconds", type=float, default=8.0)
    parser.add_argument(
        "--e6-timestamp-offset-ms",
        type=float,
        default=None,
        help=(
            "explicit calibrated (5090 realtime - E6 realtime) offset; when "
            "omitted, alignment safely uses 5090 decode-complete time"
        ),
    )
    parser.add_argument(
        "--camera-sync-max-skew-ms",
        type=float,
        default=50.0,
        help="reject a request before inference when three-camera span exceeds this",
    )
    parser.add_argument(
        "--max-observation-transport-age-ms",
        type=float,
        default=250.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--e6-ready-timeout", type=float, default=20.0)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--record-min-free-gb", type=float, default=20.0)
    parser.add_argument("--no-recording", action="store_true")
    return parser.parse_args()


def validate_checkpoint(
    checkpoint: Path,
    *,
    cam_high_mode: str,
    task_prompt: str,
) -> tuple[dict, np.ndarray]:
    profile, _norm_stats_path, state_mean_raw = validate_generic_checkpoint_contract(
        checkpoint,
        requested_cam_high_mode=cam_high_mode,
        prompt=task_prompt,
    )
    state_mean = np.asarray(state_mean_raw, dtype=np.float32)
    if state_mean.shape != (common.ACTION_DIM,):
        raise AssertionError(
            f"validated state mean changed shape unexpectedly: {state_mean.shape}"
        )
    return profile, state_mean


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.camera_sync_max_skew_ms) or args.camera_sync_max_skew_ms <= 0:
        raise ValueError("--camera-sync-max-skew-ms must be positive")
    if not np.isfinite(args.e6_history_seconds) or args.e6_history_seconds <= 0:
        raise ValueError("--e6-history-seconds must be positive")
    if args.e6_timestamp_offset_ms is not None and not np.isfinite(
        args.e6_timestamp_offset_ms
    ):
        raise ValueError("--e6-timestamp-offset-ms must be finite")
    if not np.isfinite(args.e6_max_age_ms) or args.e6_max_age_ms <= 0:
        raise ValueError("--e6-max-age-ms must be positive")
    checkpoint = args.checkpoint.resolve()
    profile, warmup_state = validate_checkpoint(
        checkpoint,
        cam_high_mode=args.cam_high_mode,
        task_prompt=args.task_prompt,
    )
    asset_id = str(profile["asset_id"])
    task_id = str(profile["task_id"])
    prompt = str(profile["prompt"])
    contract = deployment_contract_metadata(profile)
    head_preprocess = validated_head_camera_preprocess(contract)
    cam_high_selection = validate_requested_cam_high_mode(
        contract, args.cam_high_mode
    )
    cam_high_roi: CamHighRoiConfig | None = None
    if head_preprocess["normalized_ltrb"] is not None:
        cam_high_roi = CamHighRoiConfig(
            normalized_ltrb=tuple(head_preprocess["normalized_ltrb"]),
            expected_input_aspect_ratio=float(
                head_preprocess["expected_input_aspect_ratio"]
            ),
            aspect_ratio_relative_tolerance=float(
                head_preprocess["aspect_ratio_relative_tolerance"]
            ),
        )
    contract_id = str(contract["id"])
    contract_digest = profile_digest(profile)
    server_session_id = uuid.uuid4().hex
    action_horizon = _positive_contract_int(contract, "action_horizon")
    action_dim = _positive_contract_int(contract, "action_dim")
    model_action_dim = _positive_contract_int(contract, "model_action_dim")
    frequency_hz = _positive_contract_float(contract, "frequency_hz")
    if action_dim != common.ACTION_DIM:
        raise ValueError(
            "server only implements the reviewed 30-D UMI action schema: "
            f"contract={action_dim} implementation={common.ACTION_DIM}"
        )
    if model_action_dim != common.MODEL_ACTION_DIM:
        raise ValueError(
            "server only implements the padded 32-D OpenPI model schema: "
            f"contract={model_action_dim} implementation={common.MODEL_ACTION_DIM}"
        )
    expected_observation_interval_ms = 1000.0 / frequency_hz
    observation_interval_tolerance_ms = 0.45 * expected_observation_interval_ms
    expected_action_shape = (action_horizon, action_dim)

    logging.info(
        "GENERIC_CHECKPOINT_OK checkpoint=%s task=%s profile=%s asset_id=%s "
        "params_sha256=%s allowlist=disabled validation=%s",
        checkpoint,
        task_id,
        profile["profile_id"],
        asset_id,
        profile["checkpoint_params_sha256"],
        GENERIC_CHECKPOINT_VALIDATION,
    )

    # Keep globals synchronized for compatibility with diagnostics that import
    # the module, while passing every dynamic field explicitly to the config.
    common.ASSET_ID = asset_id
    common.CONFIG_NAME = str(profile["train_config_name"])
    common.DEFAULT_PROMPT = prompt

    logging.info(
        "GENERIC_CONTRACT_SELECTED task=%s profile=%s checkpoint=%s asset_id=%s "
        "prompt=%r action_mapping=%s contract=%s policy_hz=%g actions=%s",
        task_id,
        profile["profile_id"],
        checkpoint,
        asset_id,
        prompt,
        contract["action_output_basis"],
        contract_id,
        frequency_hz,
        expected_action_shape,
    )
    logging.info(
        "CAM_HIGH_PREPROCESS requested=%s effective=%s mode=%s "
        "normalized_ltrb=%s pre_resize_shape=%s",
        cam_high_selection["requested_mode"],
        cam_high_selection["effective_mode"],
        head_preprocess["mode"],
        head_preprocess["normalized_ltrb"],
        head_preprocess["pre_resize_shape_hwc"],
    )
    policy = policy_config.create_trained_policy(
        common.make_train_config(
            config_name=str(profile["train_config_name"]),
            asset_id=asset_id,
            prompt=prompt,
            discrete_state_input=bool(profile["discrete_state_input"]),
            action_horizon=action_horizon,
            frequency_hz=frequency_hz,
            cam_high_roi=cam_high_roi,
        ),
        checkpoint,
    )
    if not args.skip_warmup:
        logging.info("Running one GPU warm-up inference before opening the server")
        result = policy.infer(
            common.synthetic_observation(
                warmup_state,
                prompt=prompt,
                cam_high_shape_hwc=(
                    tuple(contract["image_shape_hwc"])
                    if cam_high_roi is not None
                    else (224, 224, 3)
                ),
            )
        )
        actions = np.asarray(result["actions"])
        if actions.shape != expected_action_shape or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"warm-up returned invalid actions: {actions.shape}")
        logging.info("Warm-up passed: actions=%s", actions.shape)
    post_load_params_sha256 = checkpoint_params_sha256(checkpoint)
    if post_load_params_sha256 != profile["checkpoint_params_sha256"]:
        raise RuntimeError(
            "checkpoint params bytes changed between validation and model load"
        )
    post_load_params_stat_sha256 = checkpoint_params_stat_sha256(checkpoint)
    if (
        post_load_params_stat_sha256
        != profile["checkpoint_params_stat_sha256"]
    ):
        raise RuntimeError(
            "checkpoint params changed between validation and model warm-up"
        )

    e6_source = E6TcpCamHigh(
        args.e6_host,
        args.e6_port,
        eye=args.e6_eye,
        history_seconds=args.e6_history_seconds,
        timestamp_offset_ms=args.e6_timestamp_offset_ms,
    )
    e6_source.start()
    deadline = time.monotonic() + args.e6_ready_timeout
    while not e6_source.status()["ready"] and time.monotonic() < deadline:
        time.sleep(0.05)
    if not e6_source.status()["ready"]:
        raise RuntimeError(f"E6 cam_h did not become ready: {e6_source.status()}")

    recorder: InferenceRecorder | None = None
    if not args.no_recording:
        recorder = InferenceRecorder(
            args.record_root,
            action_shape=expected_action_shape,
            min_free_gb=args.record_min_free_gb,
            session_metadata={
                "checkpoint": str(checkpoint),
                "asset_id": asset_id,
                "norm_stats_sha256": profile["norm_stats_sha256"],
                "checkpoint_params_sha256": profile[
                    "checkpoint_params_sha256"
                ],
                "checkpoint_params_metadata_sha256": profile[
                    "checkpoint_params_metadata_sha256"
                ],
                "task_id": task_id,
                "profile_id": profile["profile_id"],
                "prompt": prompt,
                "checkpoint_family": profile["checkpoint_family"],
                "deployment_contract_id": contract_id,
                "deployment_contract": contract,
                "deployment_profile_sha256": contract_digest,
                "contract_source": GENERIC_CONTRACT_SOURCE,
                "checkpoint_validation": GENERIC_CHECKPOINT_VALIDATION,
                "checkpoint_params_stat_sha256": profile[
                    "checkpoint_params_stat_sha256"
                ],
                "prompt_source": GENERIC_PROMPT_SOURCE,
                "server_session_id": server_session_id,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
                "frequency_hz": frequency_hz,
                "policy_host": args.host,
                "policy_port": args.port,
                "e6": e6_source.status(),
            },
        )

    policy = HeadCameraInjectingPolicy(
        policy,
        e6_source,
        max_age_ms=args.e6_max_age_ms,
        max_sync_skew_ms=args.camera_sync_max_skew_ms,
        max_observation_transport_age_ms=args.max_observation_transport_age_ms,
        recorder=recorder,
        fixed_prompt=prompt,
        cam_high_roi=cam_high_roi,
        expected_action_shape=expected_action_shape,
        expected_observation_interval_ms=expected_observation_interval_ms,
        observation_interval_tolerance_ms=observation_interval_tolerance_ms,
    )
    policy.metadata.update(
        {
            "checkpoint": str(checkpoint),
            "asset_id": asset_id,
            "norm_stats_sha256": profile["norm_stats_sha256"],
            "checkpoint_params_sha256": profile[
                "checkpoint_params_sha256"
            ],
            "checkpoint_params_metadata_sha256": profile[
                "checkpoint_params_metadata_sha256"
            ],
            "task_id": task_id,
            "profile_id": profile["profile_id"],
            "prompt": prompt,
            "checkpoint_family": profile["checkpoint_family"],
            "deployment_contract_id": contract_id,
            "deployment_contract": contract,
            "deployment_profile_sha256": contract_digest,
            "contract_source": GENERIC_CONTRACT_SOURCE,
            "checkpoint_validation": GENERIC_CHECKPOINT_VALIDATION,
            "checkpoint_params_stat_sha256": profile[
                "checkpoint_params_stat_sha256"
            ],
            "prompt_source": GENERIC_PROMPT_SOURCE,
            "server_session_id": server_session_id,
            "cam_high_mode_requested": cam_high_selection["requested_mode"],
            "cam_high_mode_effective": cam_high_selection["effective_mode"],
            "live_actuation_allowed": not args.skip_warmup,
            "recording_enabled": recorder is not None,
        }
    )
    validate_policy_contract_metadata(policy.metadata)
    logging.info(
        "Serving ws://%s:%d with cam_h=%s status=%s recording=%s",
        args.host,
        args.port,
        e6_source.endpoint,
        e6_source.status(),
        None if recorder is None else recorder.session_dir,
    )
    try:
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        ).serve_forever()
    finally:
        if recorder is not None:
            recorder.close()
        e6_source.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
