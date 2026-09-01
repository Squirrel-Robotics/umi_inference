#!/usr/bin/env python3
"""UMI live deployment contracts and checkpoint compatibility checks.

The production generic path does not use an asset/profile/weight allowlist.
It validates the checkpoint layout and 30-D normalization payload here, then
relies on model restore plus a finite ``(50, 30)`` GPU warm-up before serving.
The legacy reviewed-profile registry remains only for rollback compatibility.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "umi_live_deployment_contract_v2"
CHECKPOINT_MANIFEST_SCHEMA = "umi_checkpoint_deployment_profile_v1"
CHECKPOINT_MANIFEST_NAME = "deployment_profile.json"
PARAMS_HASH_SCHEMA = b"umi_checkpoint_params_payload_v1\0"
PROMPT_BOX_IN_OUT = "Put the object into the box, then take it out."
PROMPT_BOX_ONLY = "Put the object into the box."
PROMPT_BOX_ON_RETURN = "Put the object on the box, then return it back."
HEAD_PREPROCESS_FULL_FRAME = "full_frame_v1"
HEAD_PREPROCESS_NORMALIZED_ROI = "normalized_roi_before_resize_v1"
CAM_HIGH_MODE_AUTO = "auto"
CAM_HIGH_MODE_ROI = "roi"
CAM_HIGH_MODE_FULL = "full"
CAM_HIGH_MODE_CHOICES = (
    CAM_HIGH_MODE_AUTO,
    CAM_HIGH_MODE_ROI,
    CAM_HIGH_MODE_FULL,
)
TASK_V1_HEAD_ROI_NORMALIZED_LTRB = [0.234375, 0.35, 0.734375, 0.65]
GENERIC_CONTRACT_SOURCE = "server_fixed_generic_v1"
GENERIC_CHECKPOINT_VALIDATION = "structural_restore_warmup_v1"
GENERIC_CHECKPOINT_PARAMS_POLICY = "structural_audit_only"
GENERIC_PROMPT_SOURCE = "server_fixed_generic"


def _profile(
    *,
    profile_id: str,
    contract_id: str,
    task_id: str,
    checkpoint_family: str,
    asset_id: str,
    norm_stats_sha256: str,
    train_config_name: str,
    action_output_basis: str,
    robot_output_mapping: str,
    action_basis_config_sha256: str | None = None,
    prompt: str = PROMPT_BOX_IN_OUT,
    action_horizon: int = 50,
    frequency_hz: int | float = 10,
    execution_rate_hz: int | float | None = None,
    execution_stride: int | None = None,
    execution_first_index: int | None = None,
    head_crop_normalized_ltrb: list[float] | None = None,
    manifestless_selection: bool = False,
    checkpoint_params_policy: str = "exact_allowlist",
    params_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    contract = {
        "schema": CONTRACT_SCHEMA,
        "id": contract_id,
        "profile_id": profile_id,
        "task_id": task_id,
        "checkpoint_family": checkpoint_family,
        "asset_id": asset_id,
        "norm_stats_sha256": norm_stats_sha256,
        "train_config_name": train_config_name,
        "prompt": prompt,
        "state_dim": 30,
        "action_horizon": action_horizon,
        "action_dim": 30,
        "model_action_dim": 32,
        "frequency_hz": frequency_hz,
        "discrete_state_input": True,
        "state_schema": (
            "left_delta_eef_xyz_rot6d+right_delta_eef_xyz_rot6d+"
            "left_hand6_abs_deg+right_hand6_abs_deg"
        ),
        "action_schema": (
            "left_delta_eef_xyz_rot6d+right_delta_eef_xyz_rot6d+"
            "left_hand6_abs_deg+right_hand6_abs_deg"
        ),
        "state_builder": "adjacent_body_se3_rot6d_columns01_hand_abs_v1",
        "model_output_basis": (
            "v2" if action_output_basis == "v2_to_current" else "current"
        ),
        "input_state_basis": "robot_current_axes_unmapped",
        "eef_basis_mode": "identity",
        "action_output_basis": action_output_basis,
        "action_basis_config_sha256": action_basis_config_sha256,
        "robot_output_mapping": robot_output_mapping,
        "eef_anchor": "shared_current_observation",
        "se3_composition": "right",
        "hand_action_semantics": "absolute_degrees",
        "camera_profile": "e6_right_camera1_camera3_no_rotation_v2",
        "cam_high_eye": "right",
        "wrist_preprocess": "jpeg_bgr_to_rgb_no_rotation_v2",
        "image_shape_hwc": [480, 640, 3],
        "image_keys": [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
    }
    execution_values = (
        execution_rate_hz,
        execution_stride,
        execution_first_index,
    )
    if any(value is not None for value in execution_values):
        if any(value is None for value in execution_values):
            raise ValueError(
                "execution_rate_hz, execution_stride and "
                "execution_first_index must be provided together"
            )
        contract.update(
            {
                "execution_rate_hz": execution_rate_hz,
                "execution_stride": execution_stride,
                "execution_first_index": execution_first_index,
            }
        )
    if head_crop_normalized_ltrb is not None:
        # These fields mirror the training policy metadata.  They are absent
        # from legacy full-frame profiles so their frozen contract digests and
        # checkpoint manifests remain byte-for-byte unchanged.
        contract.update(
            {
                "cam_high_preprocess": HEAD_PREPROCESS_NORMALIZED_ROI,
                "head_crop_normalized_ltrb": list(
                    head_crop_normalized_ltrb
                ),
                "head_crop_coordinate_order": "left,top,right,bottom",
                "head_crop_expected_input_aspect": "4:3",
                "head_crop_aspect_ratio_tolerance": 0.01,
                "head_crop_applied_by": "UmiInputs-before-ResizeImages",
                "cam_high_pre_resize_shape_hwc": [144, 320, 3],
            }
        )
    profile = {
        "profile_id": profile_id,
        "task_id": task_id,
        "checkpoint_family": checkpoint_family,
        "asset_id": asset_id,
        "norm_stats_sha256": norm_stats_sha256,
        "train_config_name": train_config_name,
        "prompt": prompt,
        "discrete_state_input": True,
        "contract": contract,
    }
    if manifestless_selection:
        if checkpoint_params_policy != "family_structure":
            raise ValueError(
                "manifestless profiles must use family_structure params policy"
            )
        if not isinstance(params_metadata_sha256, str) or len(
            params_metadata_sha256
        ) != 64:
            raise ValueError(
                "manifestless profiles require a params/_METADATA SHA256"
            )
        profile.update(
            {
                "manifestless_selection": True,
                "checkpoint_params_policy": checkpoint_params_policy,
                "params_metadata_sha256": params_metadata_sha256,
            }
        )
    elif checkpoint_params_policy == GENERIC_CHECKPOINT_PARAMS_POLICY:
        profile["checkpoint_params_policy"] = checkpoint_params_policy
    elif checkpoint_params_policy != "exact_allowlist":
        raise ValueError(
            "manifest-bound profiles must retain the exact_allowlist policy"
        )
    return profile


# Only profiles with an evidence-backed live robot decoder are allowed here.
# taskumi1 is deliberately absent: its checkpoint was verified offline, but no
# matching live hardware contract was established. CX002 is a separate 14-D
# absolute-base_link stack and must never enter this UMI 30-D decoder.
DEPLOYMENT_PROFILES: tuple[dict[str, Any], ...] = (
    _profile(
        profile_id="umi_taskumi2_live_v1",
        contract_id=(
            "umi_taskumi2_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_v1"
        ),
        task_id="taskumi2",
        checkpoint_family="umi_v2",
        asset_id="umi_taskumi2_hand_pose_10hz_h50_v1",
        norm_stats_sha256=(
            "42bae3087a4e7ce8362b48f2f45202b8f334b397e0da685a93b0c2f7b1c0ed10"
        ),
        train_config_name="pi05_umi_v2_hand_pose_10hz_h50",
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
    ),
    _profile(
        profile_id="umi_taskumi3_live_v1",
        contract_id=(
            "umi_taskumi3_state30_action50x30_h50_10hz_"
            "v2out_shared_anchor_right_v1"
        ),
        task_id="taskumi3",
        checkpoint_family="umi_v3",
        asset_id="umi_taskumi3_hand_pose_v2_10hz_h50_v1",
        norm_stats_sha256=(
            "b0f08b0318c621ff4c2af6675870d28ff172fd2efa2ecc32d41616fd215486d9"
        ),
        train_config_name=(
            "pi05_umi_taskumi3_hand_pose_v2_10hz_h50_with_head_v1"
        ),
        action_output_basis="v2_to_current",
        robot_output_mapping="xr_v2_to_current_once_before_se3_decode",
        action_basis_config_sha256=(
            "2ceae98be432a19c5b8cd6e634e34b0de4199f23bdd6e7d5bec7b522d0ee3676"
        ),
    ),
    _profile(
        profile_id="umi_taskumi4_live_v1",
        contract_id=(
            "umi_taskumi4_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_v1"
        ),
        task_id="taskumi4",
        checkpoint_family="umi_v4",
        asset_id="umi_taskumi4_hand_pose_v3_10hz_h50_v1",
        norm_stats_sha256=(
            "696bac9904500bf56074ff7e90740a38c879a049cd1c85ea521e524b2f000183"
        ),
        train_config_name=(
            "pi05_umi_taskumi4_hand_pose_v3_10hz_h50_with_head_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
    ),
    _profile(
        profile_id="umi_task_v1_live_v1",
        contract_id=(
            "umi_task_v1_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_v1"
        ),
        task_id="umi_task_v1",
        checkpoint_family="umi_task_v1",
        asset_id="umi_task_v1_hand_pose_10hz_h50_masked_v1",
        norm_stats_sha256=(
            "34ac904f0b2a743532bd79cafa24aa8d1921760f8154f6b6ec22dea3a9959d2f"
        ),
        train_config_name=(
            "pi05_umi_task_v1_hand_pose_10hz_h50_masked_with_head_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=PROMPT_BOX_ONLY,
    ),
    _profile(
        profile_id="umi_task_v1_new_10hz_h50_full_live_v1",
        contract_id=(
            "umi_task_v1_new_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_full_v1"
        ),
        task_id="umi_task_v1_new",
        checkpoint_family="umi_task_v1_new_10hz_full",
        asset_id="umi_task_v1_new_hand_pose_10hz_h50_masked_v1",
        norm_stats_sha256=(
            "d9dcf361c3bc08d005eaac6c4c229e07fd34bb7bb4164008d3c1f1ba7fa6920e"
        ),
        train_config_name=(
            "pi05_umi_task_v1_new_hand_pose_10hz_h50_masked_with_head_full_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=PROMPT_BOX_ON_RETURN,
        manifestless_selection=True,
        checkpoint_params_policy="family_structure",
        params_metadata_sha256=(
            "303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"
        ),
    ),
    _profile(
        profile_id="umi_task_v1_new_10hz_h50_roi_live_v1",
        contract_id=(
            "umi_task_v1_new_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_headroi_v1"
        ),
        task_id="umi_task_v1_new",
        checkpoint_family="umi_task_v1_new_10hz_roi",
        asset_id="umi_task_v1_new_hand_pose_10hz_h50_masked_v1",
        norm_stats_sha256=(
            "d9dcf361c3bc08d005eaac6c4c229e07fd34bb7bb4164008d3c1f1ba7fa6920e"
        ),
        train_config_name=(
            "pi05_umi_task_v1_new_hand_pose_10hz_h50_masked_with_head_roi_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=PROMPT_BOX_ON_RETURN,
        head_crop_normalized_ltrb=TASK_V1_HEAD_ROI_NORMALIZED_LTRB,
        manifestless_selection=True,
        checkpoint_params_policy="family_structure",
        params_metadata_sha256=(
            "303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73"
        ),
    ),
    _profile(
        profile_id="umi_task_v3_10hz_h50_full_live_v1",
        contract_id=(
            "umi_task_v3_state30_action50x30_h50_10hz_"
            "currentout_shared_anchor_right_full_v1"
        ),
        task_id="umi_task_v3",
        checkpoint_family="umi_task_v3_10hz_h50_full",
        asset_id="umi_task_v3_hand_pose_10hz_h50_masked_v1",
        norm_stats_sha256=(
            "caad2fb146238963b4fd7815d04a7d18f6db726029c1c18544d1ab31e1308117"
        ),
        train_config_name=(
            "pi05_umi_task_v1_new_hand_pose_10hz_h50_masked_with_head_full_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=PROMPT_BOX_ON_RETURN,
        manifestless_selection=True,
        checkpoint_params_policy="family_structure",
        params_metadata_sha256=(
            "2912d15ba42fe0d742b5ac46e88804dacce69644369b4cf9b7405cf7d8fe7247"
        ),
    ),
    _profile(
        profile_id="umi_task_v1_30hz_h90_roi_live_v1",
        contract_id=(
            "umi_task_v1_state30_action90x30_h90_30hz_"
            "currentout_shared_anchor_right_headroi_v1"
        ),
        task_id="umi_task_v1",
        checkpoint_family="umi_task_v1_30hz",
        asset_id="umi_task_v1_hand_pose_30hz_h90_masked_v1",
        norm_stats_sha256=(
            "e4f5f82cd85daa7cf6260e35b17cf09db62898648d49e1fc66b3047673f5ed18"
        ),
        train_config_name=(
            "pi05_umi_task_v1_hand_pose_30hz_h90_masked_with_head_roi_v1"
        ),
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=PROMPT_BOX_ONLY,
        action_horizon=90,
        frequency_hz=30,
        execution_rate_hz=10,
        execution_stride=3,
        execution_first_index=2,
        head_crop_normalized_ltrb=TASK_V1_HEAD_ROI_NORMALIZED_LTRB,
    ),
)

# Exact model payloads promoted for live execution. These receipts bind the
# reviewed task/camera/action semantics to weights, rather than trusting a
# copied deployment_profile.json next to an arbitrary Orbax checkpoint.
# Filled from full-byte SHA256 receipts produced by
# checkpoint_params_sha256(); checkpoint step/path itself is not semantic.
REVIEWED_CHECKPOINT_PARAMS_SHA256: dict[str, frozenset[str]] = {
    "umi_taskumi2_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_v2/2000
            "60652286cbaf2b63b9b20bf381362c08d7f2c1690d3568e88b2da8ead7fcfeaf",
            # /mnt/dzq/checkpoint/umi_v2/6000
            "84d5e57fa91584491db76c76c8fbafc71a6921f9ba2660f3be19c526c070dea2",
        }
    ),
    "umi_taskumi3_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_v3/2000
            "2999586420ea405ad4a3ed97fe7274f7d3479823f0696fb0211649d44bc379a1",
            # /mnt/dzq/checkpoint/umi_v3/6000
            "b61365e8d5f6a346ec279dd8d5b6f7e370cbf6c9ad2b11469eb40028419ae87d",
            # /mnt/dzq/checkpoint/umi_v3/10000
            "be21e43acd5d27e507cd09e36dc77c5415c76b29eff12b306d84d53e050b807c",
        }
    ),
    "umi_taskumi4_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_v4/6000
            "d409158ca87bdf0de7ecb7f75d9313b1468e24fe4a89f74c08a427018fd412d1",
        }
    ),
    "umi_task_v1_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_task_v1/2000
            "4ae75ed43f02e5760bf166b06af91b0b60df2bb42191b4da2f71a12b86982f4f",
            # /mnt/dzq/checkpoint/umi_task_v1/4000
            "370ae3c1d481e8d0ab416e8554b6576d6dc6b1774e7e06946855384c471f2db0",
            # /mnt/dzq/checkpoint/umi_task_v1/6000
            "fbc1b05c56f9573b90fe65f339885fcd8b4d5673f247c6cc4830fc800d310d1e",
        }
    ),
    "umi_task_v1_new_10hz_h50_full_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_task_v1_new_10hz_full/1000
            "5186c95f320d03ad3d485ae771d98a3fafbacae8425abfeef59a9c08725cd9fb",
            # /mnt/dzq/checkpoint/umi_task_v1_new_10hz_full/2000
            "069ccab31a39f1267580e487a69a2e83426e5b621bdccdae27c56aeeb9e4b180",
        }
    ),
    "umi_task_v1_new_10hz_h50_roi_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_task_v1_new_10hz/2000
            "32725e69babd6a5deb1ed26789cd67054bbdcc78d04f56af85b1b15e0a15d879",
        }
    ),
    "umi_task_v1_30hz_h90_roi_live_v1": frozenset(
        {
            # /mnt/dzq/checkpoint/umi_task_v1_30hz/2000
            "a9c033a976aa862a55f5cd1036e1abe402330cd620cadcffd1f7a451cb390775",
            # /mnt/dzq/checkpoint/umi_task_v1_30hz/4000
            "362a001ebdaa980986963ede210e88814df86bd0ffb211a3ffbc318ad7de8ed4",
        }
    ),
}

_PROFILES_BY_ID = {profile["profile_id"]: profile for profile in DEPLOYMENT_PROFILES}
if len(_PROFILES_BY_ID) != len(DEPLOYMENT_PROFILES):
    raise RuntimeError("live profile registry contains duplicate profile IDs")
_profiles_by_asset_mutable: dict[str, list[dict[str, Any]]] = {}
for _registered_profile in DEPLOYMENT_PROFILES:
    _profiles_by_asset_mutable.setdefault(
        str(_registered_profile["asset_id"]), []
    ).append(_registered_profile)
_PROFILES_BY_ASSET: dict[str, tuple[dict[str, Any], ...]] = {
    asset_id: tuple(profiles)
    for asset_id, profiles in _profiles_by_asset_mutable.items()
}

# Backward-compatible aliases refer to the current production profile.
_DEFAULT_PROFILE = _PROFILES_BY_ID["umi_taskumi3_live_v1"]
CONTRACT_ID = str(_DEFAULT_PROFILE["contract"]["id"])
CHECKPOINT_FAMILY = str(_DEFAULT_PROFILE["checkpoint_family"])
ASSET_ID = str(_DEFAULT_PROFILE["asset_id"])
TRAIN_CONFIG_NAME = str(_DEFAULT_PROFILE["train_config_name"])
DEPLOYMENT_CONTRACT: dict[str, Any] = dict(_DEFAULT_PROFILE["contract"])


def supported_profiles() -> list[dict[str, str]]:
    return [
        {
            "profile_id": str(profile["profile_id"]),
            "task_id": str(profile["task_id"]),
            "asset_id": str(profile["asset_id"]),
            "action_output_basis": str(
                profile["contract"]["action_output_basis"]
            ),
        }
        for profile in DEPLOYMENT_PROFILES
    ]


def profile_digest(profile: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        profile["contract"],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checkpoint_params_sha256(checkpoint: Path) -> str:
    """Hash every byte of an Orbax params tree with stable file boundaries."""

    params_root = Path(checkpoint) / "params"
    if not params_root.is_dir():
        raise FileNotFoundError(f"checkpoint has no params directory: {params_root}")

    def snapshot_files() -> list[Path]:
        files: list[Path] = []
        for path in params_root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"checkpoint params must not contain symlinks: {path}")
            if path.is_file():
                files.append(path)
            elif not path.is_dir():
                raise ValueError(f"unsupported checkpoint params entry: {path}")
        return sorted(
            files,
            key=lambda item: item.relative_to(params_root).as_posix(),
        )

    files = snapshot_files()
    if not files:
        raise ValueError(f"checkpoint params tree is empty: {params_root}")
    digest = hashlib.sha256(PARAMS_HASH_SCHEMA)
    relative_names: list[str] = []
    for path in files:
        relative_name = path.relative_to(params_root).as_posix()
        relative_names.append(relative_name)
        encoded_name = relative_name.encode("utf-8")
        before = path.stat()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(before.st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if after_identity != before_identity:
            raise RuntimeError(f"checkpoint params changed while hashing: {path}")
    if [
        path.relative_to(params_root).as_posix()
        for path in snapshot_files()
    ] != relative_names:
        raise RuntimeError("checkpoint params file set changed while hashing")
    return digest.hexdigest()


def checkpoint_params_stat_sha256(checkpoint: Path) -> str:
    """Fast immutable-tree signature used to close hash-to-model-load races."""

    params_root = Path(checkpoint) / "params"
    if not params_root.is_dir():
        raise FileNotFoundError(f"checkpoint has no params directory: {params_root}")
    digest = hashlib.sha256(b"umi_checkpoint_params_stat_v1\0")
    file_count = 0
    for path in sorted(
        params_root.rglob("*"),
        key=lambda item: item.relative_to(params_root).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError(f"checkpoint params must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported checkpoint params entry: {path}")
        stat = path.stat()
        relative_name = path.relative_to(params_root).as_posix().encode("utf-8")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        for value in (
            stat.st_dev,
            stat.st_ino,
            stat.st_mode,
            stat.st_uid,
            stat.st_gid,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ):
            digest.update(int(value).to_bytes(16, "big", signed=False))
        file_count += 1
    if file_count == 0:
        raise ValueError(f"checkpoint params tree is empty: {params_root}")
    digest.update(file_count.to_bytes(8, "big"))
    return digest.hexdigest()


def validate_reviewed_params_hash(profile: Mapping[str, Any], actual: str) -> None:
    policy = profile.get("checkpoint_params_policy", "exact_allowlist")
    if policy in ("family_structure", GENERIC_CHECKPOINT_PARAMS_POLICY):
        if not isinstance(actual, str) or len(actual) != 64:
            raise ValueError(f"invalid checkpoint params SHA256: {actual!r}")
        try:
            int(actual, 16)
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint params SHA256: {actual!r}") from exc
        return
    if policy != "exact_allowlist":
        raise ValueError(
            "unsupported checkpoint params validation policy: "
            f"{policy!r}"
        )
    reviewed = REVIEWED_CHECKPOINT_PARAMS_SHA256.get(str(profile["profile_id"]))
    if reviewed is None or actual not in reviewed:
        raise ValueError(
            "checkpoint params payload is not promoted for this live profile: "
            f"profile={profile['profile_id']!r} sha256={actual!r}"
        )


def deployment_contract_metadata(
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an independent JSON/msgpack-safe contract dictionary."""

    selected = _DEFAULT_PROFILE if profile is None else profile
    contract = json.loads(json.dumps(selected["contract"]))
    validated_execution_schedule(contract)
    validated_head_camera_preprocess(contract)
    return contract


def generic_checkpoint_profile(
    *,
    asset_id: str,
    norm_stats_sha256: str,
    requested_cam_high_mode: str,
    prompt: str = PROMPT_BOX_ONLY,
) -> dict[str, Any]:
    """Build the fixed H50/10 Hz deployment contract without a whitelist.

    ``asset_id`` is used only to locate OpenPI normalization assets.  The hash
    values are audit receipts, not admission criteria.  Camera preprocessing
    and prompt cannot be inferred from Orbax, so the operator supplies them.
    """

    if requested_cam_high_mode not in (CAM_HIGH_MODE_FULL, CAM_HIGH_MODE_ROI):
        raise ValueError(
            "generic checkpoint deployment cannot infer camera preprocessing; "
            "pass explicit 'full' or 'roi'"
        )
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError("checkpoint asset_id must be a non-empty string")
    _validate_sha256("norm_stats", norm_stats_sha256)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generic task prompt must be a non-empty string")
    mode = requested_cam_high_mode
    profile = _profile(
        profile_id=f"umi_generic_h50_10hz_{mode}_v1",
        contract_id=f"umi_generic_state30_action50x30_h50_10hz_{mode}_v1",
        task_id="operator_selected_checkpoint",
        checkpoint_family="generic_h50_10hz",
        asset_id=asset_id,
        norm_stats_sha256=norm_stats_sha256,
        train_config_name=f"pi05_umi_generic_10hz_h50_{mode}_v1",
        action_output_basis="identity",
        robot_output_mapping="identity_once_before_se3_decode",
        prompt=prompt.strip(),
        head_crop_normalized_ltrb=(
            TASK_V1_HEAD_ROI_NORMALIZED_LTRB
            if mode == CAM_HIGH_MODE_ROI
            else None
        ),
        checkpoint_params_policy=GENERIC_CHECKPOINT_PARAMS_POLICY,
    )
    profile.update(
        {
            "contract_source": GENERIC_CONTRACT_SOURCE,
            "checkpoint_validation": GENERIC_CHECKPOINT_VALIDATION,
        }
    )
    profile["contract"].update(
        {
            "contract_source": GENERIC_CONTRACT_SOURCE,
            "checkpoint_validation": GENERIC_CHECKPOINT_VALIDATION,
        }
    )
    return profile


def _validate_sha256(label: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {label} SHA256: {value!r}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {label} SHA256: {value!r}") from exc
    return value


def validated_head_camera_preprocess(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the checkpoint-bound ``cam_high`` preprocessing contract.

    Legacy profiles deliberately have no ROI fields and therefore retain the
    full E6 right-eye frame.  A ROI profile must carry the complete reviewed
    training contract; partial or renamed crop metadata fails closed.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("deployment contract must be a mapping")
    roi_fields = {
        "head_crop_normalized_ltrb",
        "head_crop_coordinate_order",
        "head_crop_expected_input_aspect",
        "head_crop_aspect_ratio_tolerance",
        "head_crop_applied_by",
        "cam_high_pre_resize_shape_hwc",
    }
    mode = contract.get("cam_high_preprocess")
    if mode is None:
        unexpected = sorted(field for field in roi_fields if field in contract)
        if unexpected:
            raise ValueError(
                "full-frame cam_high contract contains orphan ROI fields: "
                f"{unexpected}"
            )
        return {
            "mode": HEAD_PREPROCESS_FULL_FRAME,
            "normalized_ltrb": None,
            "expected_input_aspect_ratio": None,
            "aspect_ratio_relative_tolerance": None,
            "pre_resize_shape_hwc": list(contract.get("image_shape_hwc", [])),
        }
    if mode != HEAD_PREPROCESS_NORMALIZED_ROI:
        raise ValueError(f"unsupported cam_high preprocessing mode: {mode!r}")

    missing = sorted(field for field in roi_fields if field not in contract)
    if missing:
        raise ValueError(f"ROI cam_high contract is incomplete: missing={missing}")
    ltrb = contract["head_crop_normalized_ltrb"]
    if not isinstance(ltrb, list) or len(ltrb) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in ltrb
    ):
        raise ValueError(
            "head_crop_normalized_ltrb must be a JSON list of four numbers"
        )
    normalized = [float(value) for value in ltrb]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("head_crop_normalized_ltrb contains NaN or Inf")
    left, top, right, bottom = normalized
    if not (
        0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
    ):
        raise ValueError("head_crop_normalized_ltrb has invalid bounds")
    if contract["head_crop_coordinate_order"] != "left,top,right,bottom":
        raise ValueError("head crop coordinate order is not LTRB")
    if contract["head_crop_expected_input_aspect"] != "4:3":
        raise ValueError("head crop input aspect contract is not 4:3")
    if contract["head_crop_applied_by"] != "UmiInputs-before-ResizeImages":
        raise ValueError("head crop is not applied at the reviewed transform stage")
    tolerance_raw = contract["head_crop_aspect_ratio_tolerance"]
    if isinstance(tolerance_raw, bool) or not isinstance(
        tolerance_raw, (int, float)
    ):
        raise ValueError("head crop aspect-ratio tolerance must be numeric")
    tolerance = float(tolerance_raw)
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("invalid head crop aspect-ratio tolerance")
    pre_resize_shape = contract["cam_high_pre_resize_shape_hwc"]
    if pre_resize_shape != [144, 320, 3]:
        raise ValueError(
            "reviewed task-v1 ROI must produce cam_high shape [144, 320, 3]"
        )
    if contract.get("image_shape_hwc") != [480, 640, 3]:
        raise ValueError(
            "reviewed task-v1 ROI requires the 640x480 E6 preprocessing frame"
        )
    return {
        "mode": HEAD_PREPROCESS_NORMALIZED_ROI,
        "normalized_ltrb": normalized,
        "expected_input_aspect_ratio": 4.0 / 3.0,
        "aspect_ratio_relative_tolerance": tolerance,
        "pre_resize_shape_hwc": list(pre_resize_shape),
    }


def validate_requested_cam_high_mode(
    contract: Mapping[str, Any],
    requested_mode: str,
) -> dict[str, str]:
    """Resolve ``auto|roi|full`` without permitting a train/deploy mismatch.

    The operator may make the intended image path explicit, but the checkpoint
    profile remains authoritative. ``roi`` never turns cropping on for a
    full-frame checkpoint, and ``full`` never bypasses a reviewed ROI.
    """

    if requested_mode not in CAM_HIGH_MODE_CHOICES:
        raise ValueError(
            "cam_high mode must be one of "
            f"{CAM_HIGH_MODE_CHOICES}; got {requested_mode!r}"
        )
    preprocess = validated_head_camera_preprocess(contract)
    effective_mode = (
        CAM_HIGH_MODE_ROI
        if preprocess["mode"] == HEAD_PREPROCESS_NORMALIZED_ROI
        else CAM_HIGH_MODE_FULL
    )
    if requested_mode != CAM_HIGH_MODE_AUTO and requested_mode != effective_mode:
        raise ValueError(
            "requested cam_high mode does not match checkpoint training: "
            f"requested={requested_mode!r} checkpoint={effective_mode!r} "
            f"profile={contract.get('profile_id')!r}"
        )
    return {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "preprocess_mode": str(preprocess["mode"]),
    }


def validate_policy_cam_high_metadata(
    metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the server's reported image path to the reviewed contract."""

    requested_mode = metadata.get("cam_high_mode_requested")
    if not isinstance(requested_mode, str):
        raise ValueError("policy metadata has no cam_high_mode_requested")
    selection = validate_requested_cam_high_mode(contract, requested_mode)
    if metadata.get("cam_high_mode_effective") != selection["effective_mode"]:
        raise ValueError(
            "policy cam_high effective mode mismatch: "
            f"expected={selection['effective_mode']!r} "
            f"actual={metadata.get('cam_high_mode_effective')!r}"
        )
    if metadata.get("cam_high_model_preprocess") != selection["preprocess_mode"]:
        raise ValueError(
            "policy cam_high preprocessing mismatch: "
            f"expected={selection['preprocess_mode']!r} "
            f"actual={metadata.get('cam_high_model_preprocess')!r}"
        )
    return selection


def validated_execution_schedule(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and strictly validate policy-to-robot execution sampling.

    Older contracts carry only ``frequency_hz`` and therefore retain their
    one-policy-step-per-control-step behavior.  A faster policy may explicitly
    declare a lower execution rate plus a stride and first sample index.  The
    selected indices must cover the final policy target, so a 90-step 30 Hz
    policy executed at 10 Hz is represented by ``2, 5, ..., 89`` rather than a
    prefix that silently drops the end of the prediction horizon.
    """

    if not isinstance(contract, Mapping):
        raise ValueError("deployment contract must be a mapping")

    action_horizon = contract.get("action_horizon")
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon <= 0
    ):
        raise ValueError(
            f"invalid action_horizon for execution schedule: {action_horizon!r}"
        )

    def positive_rate(field: str, raw: Any) -> float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"invalid {field} for execution schedule: {raw!r}")
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid {field} for execution schedule: {raw!r}")
        return value

    policy_rate_hz = positive_rate("frequency_hz", contract.get("frequency_hz"))
    execution_rate_hz = positive_rate(
        "execution_rate_hz",
        contract.get("execution_rate_hz", policy_rate_hz),
    )

    stride = contract.get("execution_stride", 1)
    first_index = contract.get("execution_first_index", 0)
    for field, value in (
        ("execution_stride", stride),
        ("execution_first_index", first_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"invalid {field} for execution schedule: {value!r}")
    if stride <= 0:
        raise ValueError(f"execution_stride must be positive: {stride!r}")
    if not 0 <= first_index < stride:
        raise ValueError(
            "execution_first_index must satisfy 0 <= first_index < "
            f"execution_stride: first={first_index!r} stride={stride!r}"
        )
    if first_index >= action_horizon:
        raise ValueError(
            "execution_first_index is outside the action horizon: "
            f"first={first_index} horizon={action_horizon}"
        )

    implied_policy_rate_hz = execution_rate_hz * stride
    if not math.isclose(
        implied_policy_rate_hz,
        policy_rate_hz,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "execution rate/stride does not reproduce policy frequency: "
            f"execution_rate_hz={execution_rate_hz} stride={stride} "
            f"policy_rate_hz={policy_rate_hz}"
        )

    indices = tuple(range(first_index, action_horizon, stride))
    if not indices or indices[-1] != action_horizon - 1:
        raise ValueError(
            "execution schedule must include the final policy target: "
            f"horizon={action_horizon} first={first_index} stride={stride} "
            f"last={None if not indices else indices[-1]}"
        )
    return {
        "policy_rate_hz": policy_rate_hz,
        "execution_rate_hz": execution_rate_hz,
        "stride": stride,
        "first_index": first_index,
        "indices": indices,
        "executable_horizon": len(indices),
    }


def checkpoint_profile_manifest(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return the explicit, human-readable live promotion marker."""

    return {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "profile_id": str(profile["profile_id"]),
        "task_id": str(profile["task_id"]),
        "asset_id": str(profile["asset_id"]),
        "norm_stats_sha256": str(profile["norm_stats_sha256"]),
        "deployment_contract_id": str(profile["contract"]["id"]),
        "deployment_profile_sha256": profile_digest(profile),
    }


def discover_checkpoint_norm_stats(checkpoint: Path) -> tuple[str, Path]:
    """Find the checkpoint's sole normalization asset."""

    assets_root = Path(checkpoint) / "assets"
    candidates = sorted(
        path for path in assets_root.glob("*/norm_stats.json") if path.is_file()
    )
    if len(candidates) != 1:
        rendered = [str(path) for path in candidates]
        raise ValueError(
            "checkpoint must contain exactly one assets/<asset_id>/norm_stats.json; "
            f"found={rendered}"
        )
    norm_stats_path = candidates[0]
    return norm_stats_path.parent.name, norm_stats_path


def select_profile(
    checkpoint: Path,
    asset_id: str,
    *,
    requested_cam_high_mode: str = CAM_HIGH_MODE_AUTO,
    norm_stats_sha256: str | None = None,
) -> dict[str, Any]:
    """Select a reviewed profile from a manifest or explicit family mode.

    Existing deployments retain their exact per-checkpoint manifest. Reviewed
    training families may omit it when ``asset_id + norm SHA + full|roi`` maps
    to exactly one profile. ``auto`` is intentionally rejected for those
    families because current Orbax metadata cannot distinguish ROI from full.
    """

    if requested_cam_high_mode not in CAM_HIGH_MODE_CHOICES:
        raise ValueError(
            "cam_high mode must be one of "
            f"{CAM_HIGH_MODE_CHOICES}; got {requested_cam_high_mode!r}"
        )
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / CHECKPOINT_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            raw_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid checkpoint deployment manifest: {manifest_path}"
            ) from exc
        if not isinstance(raw_manifest, Mapping):
            raise ValueError("checkpoint deployment manifest must be a JSON object")
        profile_id = raw_manifest.get("profile_id")
        profile = _PROFILES_BY_ID.get(profile_id)
        if profile is None:
            supported = ", ".join(sorted(_PROFILES_BY_ID))
            raise ValueError(
                "checkpoint manifest names no reviewed UMI live profile: "
                f"profile_id={profile_id!r}; supported=[{supported}]"
            )
        expected_manifest = checkpoint_profile_manifest(profile)
        mismatches = [
            f"{key}: expected={expected!r} actual={raw_manifest.get(key)!r}"
            for key, expected in expected_manifest.items()
            if raw_manifest.get(key) != expected
        ]
        extra = sorted(set(raw_manifest) - set(expected_manifest))
        if extra:
            mismatches.append(f"unexpected keys={extra}")
        if mismatches:
            raise ValueError(
                "checkpoint deployment manifest mismatch: "
                + "; ".join(mismatches)
            )
        if asset_id != profile["asset_id"]:
            raise ValueError(
                "checkpoint asset does not match its deployment manifest: "
                f"manifest={profile['asset_id']!r} discovered={asset_id!r}"
            )
    else:
        if requested_cam_high_mode == CAM_HIGH_MODE_AUTO:
            raise ValueError(
                "checkpoint has no deployment_profile.json and its camera "
                "preprocessing cannot be inferred; pass explicit 'full' or "
                f"'roi': {checkpoint}"
            )
        if not isinstance(norm_stats_sha256, str):
            raise ValueError(
                "manifestless profile selection requires the norm_stats SHA256"
            )
        candidates: list[dict[str, Any]] = []
        for candidate in _PROFILES_BY_ASSET.get(asset_id, ()):
            if candidate.get("manifestless_selection") is not True:
                continue
            if candidate["norm_stats_sha256"] != norm_stats_sha256:
                continue
            try:
                validate_requested_cam_high_mode(
                    candidate["contract"], requested_cam_high_mode
                )
            except ValueError:
                continue
            candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError(
                "checkpoint does not match exactly one reviewed manifestless "
                "profile: "
                f"asset_id={asset_id!r} norm_stats_sha256="
                f"{norm_stats_sha256!r} cam_high_mode="
                f"{requested_cam_high_mode!r} candidates="
                f"{[item['profile_id'] for item in candidates]}"
            )
        profile = candidates[0]
    validate_requested_cam_high_mode(
        profile["contract"], requested_cam_high_mode
    )
    # Headless taskumi3 reused the three-camera norm asset, so asset ID alone
    # cannot distinguish it. Keep this explicit second defense even though all
    # live checkpoints now also require a reviewed manifest.
    if "nohead" in str(checkpoint).lower():
        raise ValueError(
            "headless checkpoint is not compatible with the three-camera live "
            "profile; use a distinct reviewed profile/asset ID"
        )
    return profile


def preflight_checkpoint_cam_high_mode(
    checkpoint: Path,
    requested_mode: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Check only manifest/asset camera semantics, without hashing weights."""

    checkpoint = Path(checkpoint)
    asset_id, norm_stats_path = discover_checkpoint_norm_stats(checkpoint)
    norm_stats_sha256 = payload_sha256(norm_stats_path.read_bytes())
    profile = select_profile(
        checkpoint,
        asset_id,
        requested_cam_high_mode=requested_mode,
        norm_stats_sha256=norm_stats_sha256,
    )
    selection = validate_requested_cam_high_mode(
        profile["contract"], requested_mode
    )
    return profile, selection


def validate_norm_stats_payload(raw: Mapping[str, Any]) -> list[float]:
    """Validate every statistic used by 30-D quantile normalization."""

    norm_stats = raw.get("norm_stats")
    if not isinstance(norm_stats, Mapping):
        raise ValueError("norm_stats.json has no norm_stats mapping")
    validated: dict[str, dict[str, list[float]]] = {}
    for stream in ("state", "actions"):
        stream_raw = norm_stats.get(stream)
        if not isinstance(stream_raw, Mapping):
            raise ValueError(f"norm_stats.json has no {stream} mapping")
        validated[stream] = {}
        for statistic in ("mean", "std", "q01", "q99"):
            values_raw = stream_raw.get(statistic)
            if not isinstance(values_raw, list) or len(values_raw) != 30:
                length = None if not isinstance(values_raw, list) else len(values_raw)
                raise ValueError(
                    "bad norm stats shape: "
                    f"{stream}.{statistic}=({length},), expected=(30,)"
                )
            if any(isinstance(value, bool) for value in values_raw):
                raise ValueError(f"invalid boolean norm stat: {stream}.{statistic}")
            try:
                values = [float(value) for value in values_raw]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"non-numeric norm stat: {stream}.{statistic}"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"norm stats contain NaN or Inf: {stream}.{statistic}"
                )
            validated[stream][statistic] = values
        if any(value < 0.0 for value in validated[stream]["std"]):
            raise ValueError(f"norm stats contain negative std: {stream}")
        if any(
            high < low
            for low, high in zip(
                validated[stream]["q01"],
                validated[stream]["q99"],
                strict=True,
            )
        ):
            raise ValueError(f"norm stats have q99 < q01: {stream}")
    return validated["state"]["mean"]


def preflight_generic_checkpoint_cam_high_mode(
    checkpoint: Path,
    requested_mode: str,
    *,
    prompt: str = PROMPT_BOX_ONLY,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate generic checkpoint inputs without hashing/loading the model."""

    checkpoint = Path(checkpoint)
    if requested_mode not in (CAM_HIGH_MODE_FULL, CAM_HIGH_MODE_ROI):
        raise ValueError(
            "generic checkpoint deployment requires explicit 'full' or 'roi'"
        )
    asset_id, norm_stats_path = discover_checkpoint_norm_stats(checkpoint)
    required = (
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params" / "_METADATA",
        norm_stats_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete inference checkpoint; missing: {missing}")
    norm_stats_bytes = norm_stats_path.read_bytes()
    try:
        raw = json.loads(norm_stats_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid norm_stats JSON: {norm_stats_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("norm_stats.json must contain a JSON object")
    validate_norm_stats_payload(raw)
    profile = generic_checkpoint_profile(
        asset_id=asset_id,
        norm_stats_sha256=payload_sha256(norm_stats_bytes),
        requested_cam_high_mode=requested_mode,
        prompt=prompt,
    )
    selection = validate_requested_cam_high_mode(
        profile["contract"], requested_mode
    )
    return profile, selection


def validate_generic_checkpoint_contract(
    checkpoint: Path,
    *,
    requested_cam_high_mode: str,
    prompt: str = PROMPT_BOX_ONLY,
) -> tuple[dict[str, Any], Path, list[float]]:
    """Validate a checkpoint structurally, without any allowlist lookup."""

    checkpoint = Path(checkpoint)
    profile, _selection = preflight_generic_checkpoint_cam_high_mode(
        checkpoint,
        requested_cam_high_mode,
        prompt=prompt,
    )
    _asset_id, norm_stats_path = discover_checkpoint_norm_stats(checkpoint)
    raw = json.loads(norm_stats_path.read_bytes())
    state_mean = validate_norm_stats_payload(raw)
    params_metadata_path = checkpoint / "params" / "_METADATA"
    actual_params_metadata_sha256 = payload_sha256(
        params_metadata_path.read_bytes()
    )
    actual_params_sha256 = checkpoint_params_sha256(checkpoint)
    validate_reviewed_params_hash(profile, actual_params_sha256)
    selected = dict(profile)
    selected["checkpoint_params_sha256"] = actual_params_sha256
    selected["checkpoint_params_metadata_sha256"] = (
        actual_params_metadata_sha256
    )
    selected["checkpoint_params_stat_sha256"] = checkpoint_params_stat_sha256(
        checkpoint
    )
    return selected, norm_stats_path, state_mean


def validate_checkpoint_contract(
    checkpoint: Path,
    *,
    requested_cam_high_mode: str = CAM_HIGH_MODE_AUTO,
) -> tuple[dict[str, Any], Path, list[float]]:
    """Validate and resolve a checkpoint without JAX, ROS, cameras or OpenPI."""

    checkpoint = Path(checkpoint)
    asset_id, norm_stats_path = discover_checkpoint_norm_stats(checkpoint)
    norm_stats_bytes = norm_stats_path.read_bytes()
    actual_norm_stats_sha256 = payload_sha256(norm_stats_bytes)
    profile = select_profile(
        checkpoint,
        asset_id,
        requested_cam_high_mode=requested_cam_high_mode,
        norm_stats_sha256=actual_norm_stats_sha256,
    )
    required = (
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params" / "_METADATA",
        norm_stats_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete inference checkpoint; missing: {missing}")
    raw = json.loads(norm_stats_bytes)
    if actual_norm_stats_sha256 != profile["norm_stats_sha256"]:
        raise ValueError(
            "checkpoint norm_stats hash does not match its reviewed profile: "
            f"expected={profile['norm_stats_sha256']!r} "
            f"actual={actual_norm_stats_sha256!r}"
        )
    state_mean = validate_norm_stats_payload(raw)
    params_metadata_path = checkpoint / "params" / "_METADATA"
    actual_params_metadata_sha256 = payload_sha256(
        params_metadata_path.read_bytes()
    )
    expected_params_metadata_sha256 = profile.get("params_metadata_sha256")
    if (
        expected_params_metadata_sha256 is not None
        and actual_params_metadata_sha256 != expected_params_metadata_sha256
    ):
        raise ValueError(
            "checkpoint params structure does not match its reviewed family: "
            f"expected={expected_params_metadata_sha256!r} "
            f"actual={actual_params_metadata_sha256!r}"
        )
    actual_params_sha256 = checkpoint_params_sha256(checkpoint)
    validate_reviewed_params_hash(profile, actual_params_sha256)
    selected = dict(profile)
    selected["checkpoint_params_sha256"] = actual_params_sha256
    selected["checkpoint_params_metadata_sha256"] = (
        actual_params_metadata_sha256
    )
    selected["checkpoint_params_stat_sha256"] = checkpoint_params_stat_sha256(
        checkpoint
    )
    return selected, norm_stats_path, state_mean


def validate_policy_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_contract_id: str | None = None,
) -> dict[str, Any]:
    """Select a local profile and exact-match all server contract semantics."""

    raw_contract = metadata.get("deployment_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("policy metadata has no deployment_contract mapping")
    if raw_contract.get("contract_source") == GENERIC_CONTRACT_SOURCE:
        return validate_generic_policy_contract_metadata(
            metadata,
            expected_contract_id=expected_contract_id,
        )
    profile_id = raw_contract.get("profile_id")
    profile = _PROFILES_BY_ID.get(profile_id)
    if profile is None:
        raise ValueError(f"unknown live deployment profile: {profile_id!r}")
    expected_contract = profile["contract"]
    actual_id = metadata.get("deployment_contract_id")
    registered_id = expected_contract["id"]
    if expected_contract_id is not None and registered_id != expected_contract_id:
        raise ValueError(
            "requested deployment contract does not identify this profile: "
            f"requested={expected_contract_id!r} registered={registered_id!r}"
        )
    if actual_id != registered_id:
        raise ValueError(
            "deployment contract id mismatch: "
            f"expected={registered_id!r} actual={actual_id!r}"
        )
    mismatches = [
        f"{key}: expected={expected!r} actual={raw_contract.get(key)!r}"
        for key, expected in expected_contract.items()
        if raw_contract.get(key) != expected
    ]
    extra = sorted(set(raw_contract) - set(expected_contract))
    if extra:
        mismatches.append(f"unexpected contract keys={extra}")
    if mismatches:
        raise ValueError("deployment contract mismatch: " + "; ".join(mismatches))
    validated_execution_schedule(raw_contract)
    validated_head_camera_preprocess(raw_contract)
    validate_policy_cam_high_metadata(metadata, raw_contract)
    for key in (
        "profile_id",
        "task_id",
        "asset_id",
        "norm_stats_sha256",
        "checkpoint_family",
        "prompt",
    ):
        expected = profile[key]
        if metadata.get(key) != expected:
            raise ValueError(
                f"policy metadata {key} mismatch: "
                f"expected={expected!r} actual={metadata.get(key)!r}"
            )
    expected_digest = profile_digest(profile)
    if metadata.get("deployment_profile_sha256") != expected_digest:
        raise ValueError(
            "deployment profile digest mismatch: "
            f"expected={expected_digest!r} "
            f"actual={metadata.get('deployment_profile_sha256')!r}"
        )
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("/"):
        raise ValueError(f"policy metadata checkpoint is not absolute: {checkpoint!r}")
    params_sha256 = metadata.get("checkpoint_params_sha256")
    if not isinstance(params_sha256, str):
        raise ValueError("policy metadata has no checkpoint params SHA256")
    validate_reviewed_params_hash(profile, params_sha256)
    expected_params_metadata_sha256 = profile.get("params_metadata_sha256")
    if expected_params_metadata_sha256 is not None:
        actual_params_metadata_sha256 = metadata.get(
            "checkpoint_params_metadata_sha256"
        )
        if actual_params_metadata_sha256 != expected_params_metadata_sha256:
            raise ValueError(
                "policy checkpoint params structure mismatch: "
                f"expected={expected_params_metadata_sha256!r} "
                f"actual={actual_params_metadata_sha256!r}"
            )
    return dict(raw_contract)


def validate_generic_policy_contract_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_contract_id: str | None = None,
) -> dict[str, Any]:
    """Exact-match server metadata against the local no-allowlist template."""

    raw_contract = metadata.get("deployment_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("policy metadata has no deployment_contract mapping")
    if metadata.get("contract_source") != GENERIC_CONTRACT_SOURCE:
        raise ValueError("policy metadata is not from the fixed generic contract")
    if metadata.get("checkpoint_validation") != GENERIC_CHECKPOINT_VALIDATION:
        raise ValueError("policy metadata has no structural restore/warm-up receipt")
    requested_mode = metadata.get("cam_high_mode_requested")
    if requested_mode not in (CAM_HIGH_MODE_FULL, CAM_HIGH_MODE_ROI):
        raise ValueError(
            "generic policy metadata must declare explicit full or roi mode"
        )
    asset_id = metadata.get("asset_id")
    norm_stats_sha256 = metadata.get("norm_stats_sha256")
    prompt = metadata.get("prompt")
    expected_profile = generic_checkpoint_profile(
        asset_id=asset_id,
        norm_stats_sha256=norm_stats_sha256,
        requested_cam_high_mode=requested_mode,
        prompt=prompt,
    )
    expected_contract = expected_profile["contract"]
    registered_id = expected_contract["id"]
    if expected_contract_id is not None and registered_id != expected_contract_id:
        raise ValueError(
            "requested deployment contract mismatch: "
            f"requested={expected_contract_id!r} generic={registered_id!r}"
        )
    if metadata.get("deployment_contract_id") != registered_id:
        raise ValueError(
            "deployment contract id mismatch: "
            f"expected={registered_id!r} "
            f"actual={metadata.get('deployment_contract_id')!r}"
        )
    mismatches = [
        f"{key}: expected={expected!r} actual={raw_contract.get(key)!r}"
        for key, expected in expected_contract.items()
        if raw_contract.get(key) != expected
    ]
    extra = sorted(set(raw_contract) - set(expected_contract))
    if extra:
        mismatches.append(f"unexpected contract keys={extra}")
    if mismatches:
        raise ValueError("generic deployment contract mismatch: " + "; ".join(mismatches))
    validated_execution_schedule(raw_contract)
    validated_head_camera_preprocess(raw_contract)
    validate_policy_cam_high_metadata(metadata, raw_contract)
    for key in (
        "profile_id",
        "task_id",
        "asset_id",
        "norm_stats_sha256",
        "checkpoint_family",
        "prompt",
    ):
        expected = expected_profile[key]
        if metadata.get(key) != expected:
            raise ValueError(
                f"generic policy metadata {key} mismatch: "
                f"expected={expected!r} actual={metadata.get(key)!r}"
            )
    expected_digest = profile_digest(expected_profile)
    if metadata.get("deployment_profile_sha256") != expected_digest:
        raise ValueError(
            "generic deployment contract digest mismatch: "
            f"expected={expected_digest!r} "
            f"actual={metadata.get('deployment_profile_sha256')!r}"
        )
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("/"):
        raise ValueError(f"policy metadata checkpoint is not absolute: {checkpoint!r}")
    for key in (
        "checkpoint_params_sha256",
        "checkpoint_params_metadata_sha256",
        "checkpoint_params_stat_sha256",
    ):
        _validate_sha256(key, metadata.get(key))
    if metadata.get("prompt_source") != GENERIC_PROMPT_SOURCE:
        raise ValueError(
            "generic policy server does not enforce its fixed task prompt"
        )
    return dict(raw_contract)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a UMI checkpoint without a checkpoint allowlist"
    )
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--print-manifest", metavar="PROFILE_ID")
    parser.add_argument(
        "--check-cam-high-mode",
        choices=CAM_HIGH_MODE_CHOICES,
        help=(
            "lightweight structural check for explicit roi|full; does not "
            "hash or load model parameters"
        ),
    )
    parser.add_argument(
        "--generic-prompt",
        default=PROMPT_BOX_ONLY,
        help="operator-selected prompt used by the fixed generic contract",
    )
    parser.add_argument(
        "--print-params-sha256",
        metavar="CHECKPOINT",
        type=Path,
        help="hash every file byte in CHECKPOINT/params without loading a model",
    )
    args = parser.parse_args()
    if args.list_profiles:
        print(json.dumps(supported_profiles(), indent=2, ensure_ascii=False))
        return
    if args.print_manifest is not None:
        profile = _PROFILES_BY_ID.get(args.print_manifest)
        if profile is None:
            parser.error(f"unknown profile ID: {args.print_manifest}")
        print(
            json.dumps(
                checkpoint_profile_manifest(profile),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.print_params_sha256 is not None:
        checkpoint = args.print_params_sha256.resolve()
        print(checkpoint_params_sha256(checkpoint))
        return
    if args.checkpoint is None:
        parser.error(
            "checkpoint is required unless --list-profiles or --print-manifest is used"
        )
    checkpoint = args.checkpoint.resolve()
    if args.check_cam_high_mode is not None:
        try:
            profile, selection = preflight_generic_checkpoint_cam_high_mode(
                checkpoint,
                args.check_cam_high_mode,
                prompt=args.generic_prompt,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            "GENERIC_CHECKPOINT_PREFLIGHT_OK "
            f"checkpoint={checkpoint} profile={profile['profile_id']} "
            "allowlist=disabled "
            f"requested={selection['requested_mode']} "
            f"effective={selection['effective_mode']} "
            f"preprocess={selection['preprocess_mode']}"
        )
        return
    profile, norm_stats_path, _state_mean = validate_checkpoint_contract(checkpoint)
    contract = profile["contract"]
    print(
        "CHECKPOINT_PROFILE_OK "
        f"checkpoint={checkpoint} task={profile['task_id']} "
        f"profile={profile['profile_id']} asset_id={profile['asset_id']} "
        f"prompt={json.dumps(profile['prompt'])} "
        f"action_mapping={contract['action_output_basis']}+"
        f"{contract['eef_anchor']}+{contract['se3_composition']} "
        f"norm_stats={norm_stats_path} "
        f"params_sha256={profile['checkpoint_params_sha256']}"
    )


if __name__ == "__main__":
    main()
