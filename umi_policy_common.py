"""Exact OpenPI policy/data contract used by the UMI pi0.5 training run."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from cam_high_roi import CamHighRoiConfig, crop_cam_high_roi
from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as training_config


CONFIG_NAME = "pi05_umi_latest100_delta_eef_rot6d_hand30_10hz_h50"
ASSET_ID = "umi_latest100_delta_eef_rot6d_hand30_10hz_h50"
DEFAULT_PROMPT = "Put the object into the box, then take it out."
ACTION_DIM = 30
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 50
DEFAULT_FREQUENCY_HZ = 10


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"image must have 3 axes, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        # Training-side LeRobot images are floats in [0, 1]. Deployment may send uint8.
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 + 1e-6 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image.astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Convert robot observations to the exact schema used during training."""

    cam_high_roi: CamHighRoiConfig | None = None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = data["images"]
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"state must be ({ACTION_DIM},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or Inf")

        cam_high = _to_hwc_uint8(images["cam_high"])
        if self.cam_high_roi is not None:
            cam_high = crop_cam_high_roi(
                cam_high, config=self.cam_high_roi
            ).image

        output: dict[str, Any] = {
            "image": {
                "base_0_rgb": cam_high,
                "left_wrist_0_rgb": _to_hwc_uint8(images["cam_left_wrist"]),
                "right_wrist_0_rgb": _to_hwc_uint8(images["cam_right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }
        if "prompt" in data:
            prompt = data["prompt"]
            output["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        return output


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = np.asarray(data["actions"][..., :ACTION_DIM], dtype=np.float32)
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class UmiDataConfig(training_config.DataConfigFactory):
    repo_id: str = "local/umi_latest100"
    assets: training_config.AssetsConfig = dataclasses.field(
        default_factory=lambda: training_config.AssetsConfig(asset_id=ASSET_ID)
    )
    default_prompt: str = DEFAULT_PROMPT
    action_horizon: int = ACTION_HORIZON
    cam_high_roi: CamHighRoiConfig | None = None

    def create(self, assets_dirs, model_config):  # noqa: ANN001
        if model_config.action_horizon != self.action_horizon:
            raise ValueError(
                "checkpoint action horizon does not match the data contract: "
                f"model={model_config.action_horizon} "
                f"contract={self.action_horizon}"
            )
        if model_config.action_dim != MODEL_ACTION_DIM:
            raise ValueError("checkpoint requires padded model action_dim=32")
        return training_config.DataConfig(
            repo_id=self.repo_id,
            asset_id=self.assets.asset_id,
            data_transforms=transforms.Group(
                inputs=[UmiInputs(cam_high_roi=self.cam_high_roi)],
                outputs=[UmiOutputs()],
            ),
            model_transforms=training_config.ModelTransformFactory(
                default_prompt=self.default_prompt
            )(model_config),
            use_quantile_norm=True,
            # Each row was prechunked to (H, 30); no future-row gathering is allowed.
            action_sequence_keys=(),
        )


def make_train_config(
    *,
    config_name: str | None = None,
    asset_id: str | None = None,
    prompt: str | None = None,
    discrete_state_input: bool = True,
    action_horizon: int | None = None,
    frequency_hz: float | None = None,
    cam_high_roi: CamHighRoiConfig | None = None,
) -> training_config.TrainConfig:
    """Construct the inference-equivalent of the original training config."""

    selected_action_horizon = (
        ACTION_HORIZON if action_horizon is None else action_horizon
    )
    if (
        isinstance(selected_action_horizon, bool)
        or not isinstance(selected_action_horizon, int)
        or selected_action_horizon <= 0
    ):
        raise ValueError(
            "action_horizon must be a positive integer, got "
            f"{selected_action_horizon!r}"
        )
    selected_frequency = (
        DEFAULT_FREQUENCY_HZ if frequency_hz is None else frequency_hz
    )
    if isinstance(selected_frequency, bool):
        raise ValueError(
            "frequency_hz must be finite and positive, got "
            f"{selected_frequency!r}"
        )
    try:
        frequency_number = float(selected_frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "frequency_hz must be finite and positive, got "
            f"{selected_frequency!r}"
        ) from exc
    if not np.isfinite(frequency_number) or frequency_number <= 0.0:
        raise ValueError(
            "frequency_hz must be finite and positive, got "
            f"{selected_frequency!r}"
        )
    selected_frequency_hz: int | float = (
        int(frequency_number) if frequency_number.is_integer() else frequency_number
    )

    selected_config_name = CONFIG_NAME if config_name is None else config_name
    selected_asset_id = ASSET_ID if asset_id is None else asset_id
    selected_prompt = DEFAULT_PROMPT if prompt is None else prompt

    return training_config.TrainConfig(
        name=selected_config_name,
        exp_name="inference",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_ACTION_DIM,
            action_horizon=selected_action_horizon,
            discrete_state_input=discrete_state_input,
        ),
        data=UmiDataConfig(
            assets=training_config.AssetsConfig(asset_id=selected_asset_id),
            default_prompt=selected_prompt,
            action_horizon=selected_action_horizon,
            cam_high_roi=cam_high_roi,
        ),
        wandb_enabled=False,
        policy_metadata={
            "schema": "umi-lerobot-delta-eef-rot6d-hand30",
            "frequency_hz": selected_frequency_hz,
            "action_horizon": selected_action_horizon,
            "action_dim": ACTION_DIM,
            "prompt": selected_prompt,
            "eef_anchor": "all horizon steps share the current observation pose",
        },
    )


def synthetic_observation(
    state: np.ndarray | None = None,
    *,
    prompt: str | None = None,
    cam_high_shape_hwc: tuple[int, int, int] = (224, 224, 3),
) -> dict[str, Any]:
    if state is None:
        state = np.concatenate(
            [
                np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
                np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32),
                np.zeros(12, dtype=np.float32),
            ]
        )
    if (
        not isinstance(cam_high_shape_hwc, tuple)
        or len(cam_high_shape_hwc) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in cam_high_shape_hwc
        )
        or cam_high_shape_hwc[2] != 3
    ):
        raise ValueError(
            "cam_high_shape_hwc must be a positive RGB HWC tuple, got "
            f"{cam_high_shape_hwc!r}"
        )
    cam_high = np.full(cam_high_shape_hwc, 127, dtype=np.uint8)
    wrist = np.full((224, 224, 3), 127, dtype=np.uint8)
    return {
        "images": {
            "cam_high": cam_high,
            "cam_left_wrist": wrist.copy(),
            "cam_right_wrist": wrist.copy(),
        },
        "state": np.asarray(state, dtype=np.float32),
        "prompt": DEFAULT_PROMPT if prompt is None else prompt,
    }
