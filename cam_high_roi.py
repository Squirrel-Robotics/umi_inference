"""Pure NumPy preprocessing for the task-v1 ``cam_high`` ROI.

The ROI is expressed as normalized ``(left, top, right, bottom)`` coordinates
so the same contract works for every 4:3 E6 frame size.  This module is kept
independent of OpenPI and the recorder so both can call the exact same code.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Final, Sequence

import numpy as np


TASK_V1_CAM_HIGH_NORMALIZED_LTRB: Final[tuple[float, float, float, float]] = (
    0.234375,
    0.35,
    0.734375,
    0.65,
)
TASK_V1_CAM_HIGH_INPUT_ASPECT_RATIO: Final[float] = 4.0 / 3.0
TASK_V1_CAM_HIGH_ASPECT_RATIO_REL_TOLERANCE: Final[float] = 0.01
ROI_ROUNDING_CONTRACT: Final[str] = (
    "floor(left/top + 1e-9), ceil(right/bottom - 1e-9)"
)


def _validated_ltrb(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(
            "normalized_ltrb must contain four numbers in "
            "(left, top, right, bottom) order"
        )
    try:
        left, top, right, bottom = (float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalized_ltrb must contain finite numbers") from exc
    if not all(math.isfinite(component) for component in (left, top, right, bottom)):
        raise ValueError("normalized_ltrb must contain finite numbers")
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise ValueError(
            "normalized_ltrb must satisfy "
            "0 <= left < right <= 1 and 0 <= top < bottom <= 1"
        )
    return left, top, right, bottom


@dataclasses.dataclass(frozen=True)
class CamHighRoiConfig:
    """Validated, JSON-serializable ROI contract for ``cam_high`` only."""

    normalized_ltrb: tuple[float, float, float, float] = (
        TASK_V1_CAM_HIGH_NORMALIZED_LTRB
    )
    expected_input_aspect_ratio: float = TASK_V1_CAM_HIGH_INPUT_ASPECT_RATIO
    aspect_ratio_relative_tolerance: float = (
        TASK_V1_CAM_HIGH_ASPECT_RATIO_REL_TOLERANCE
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "normalized_ltrb", _validated_ltrb(self.normalized_ltrb)
        )
        expected = float(self.expected_input_aspect_ratio)
        tolerance = float(self.aspect_ratio_relative_tolerance)
        if not math.isfinite(expected) or expected <= 0.0:
            raise ValueError("expected_input_aspect_ratio must be finite and positive")
        if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
            raise ValueError(
                "aspect_ratio_relative_tolerance must be finite and in [0, 1)"
            )
        object.__setattr__(self, "expected_input_aspect_ratio", expected)
        object.__setattr__(self, "aspect_ratio_relative_tolerance", tolerance)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "cam_high_normalized_roi_v1",
            "camera": "cam_high",
            "coordinate_order": "left,top,right,bottom",
            "normalized_ltrb": list(self.normalized_ltrb),
            "expected_input_aspect_ratio": self.expected_input_aspect_ratio,
            "aspect_ratio_relative_tolerance": (
                self.aspect_ratio_relative_tolerance
            ),
            "rounding": ROI_ROUNDING_CONTRACT,
        }


@dataclasses.dataclass(frozen=True)
class CamHighRoiBounds:
    """Resolved half-open pixel bounds for one validated source frame."""

    input_height: int
    input_width: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def output_height(self) -> int:
        return self.bottom - self.top

    @property
    def output_width(self) -> int:
        return self.right - self.left

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.top, self.bottom), slice(self.left, self.right)

    def metadata(self) -> dict[str, Any]:
        return {
            "input_shape_hwc": [self.input_height, self.input_width, 3],
            "pixel_bounds_ltrb": [self.left, self.top, self.right, self.bottom],
            "pixel_bounds_are_half_open": True,
            "output_shape_hwc": [self.output_height, self.output_width, 3],
        }


@dataclasses.dataclass(frozen=True)
class CamHighRoiResult:
    """Cropped image plus the complete preprocessing proof for recording."""

    image: np.ndarray
    bounds: CamHighRoiBounds
    config: CamHighRoiConfig
    source_layout: str
    source_dtype: str

    def metadata(self) -> dict[str, Any]:
        return {
            **self.config.metadata(),
            **self.bounds.metadata(),
            "source_layout": self.source_layout,
            "source_dtype": self.source_dtype,
            "output_dtype": str(self.image.dtype),
            "output_c_contiguous": bool(self.image.flags.c_contiguous),
        }


TASK_V1_CAM_HIGH_ROI: Final[CamHighRoiConfig] = CamHighRoiConfig()


def to_hwc_uint8(image: np.ndarray) -> tuple[np.ndarray, str, str]:
    """Convert an RGB HWC/CHW image to contiguous HWC uint8.

    Floating-point inputs in ``[0, 1]`` use the training-side ``* 255``
    conversion.  Other numeric inputs are clipped directly to ``[0, 255]``.
    The returned layout and dtype describe the source and are useful in an
    inference record.
    """

    source = np.asarray(image)
    source_dtype = str(source.dtype)
    if source.ndim != 3:
        raise ValueError(f"cam_high must have three axes, got {source.shape}")
    if source.size == 0:
        raise ValueError("cam_high must not be empty")
    if source.shape[-1] == 3:
        source_layout = "HWC"
        hwc = source
    elif source.shape[0] == 3:
        source_layout = "CHW"
        hwc = np.transpose(source, (1, 2, 0))
    else:
        raise ValueError(
            "cam_high must be RGB in HWC or CHW layout, "
            f"got {source.shape}"
        )
    if np.issubdtype(hwc.dtype, np.complexfloating) or not (
        np.issubdtype(hwc.dtype, np.integer)
        or np.issubdtype(hwc.dtype, np.floating)
    ):
        raise TypeError(f"cam_high must have a real numeric dtype, got {hwc.dtype}")
    if not np.all(np.isfinite(hwc)):
        raise ValueError("cam_high contains NaN or Inf")

    if np.issubdtype(hwc.dtype, np.floating):
        scale = 255.0 if float(np.max(hwc)) <= 1.0 + 1e-6 else 1.0
        converted = np.clip(hwc * scale, 0.0, 255.0).astype(np.uint8)
    elif hwc.dtype == np.uint8:
        converted = hwc
    else:
        converted = np.clip(hwc, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(converted), source_layout, source_dtype


def resolve_cam_high_roi_bounds(
    image_shape_hwc: Sequence[int],
    *,
    config: CamHighRoiConfig = TASK_V1_CAM_HIGH_ROI,
) -> CamHighRoiBounds:
    """Resolve the normalized contract to half-open integer pixel bounds."""

    if len(image_shape_hwc) != 3:
        raise ValueError(f"image_shape_hwc must have three axes, got {image_shape_hwc}")
    height, width, channels = (int(value) for value in image_shape_hwc)
    if height <= 0 or width <= 0 or channels != 3:
        raise ValueError(
            "image_shape_hwc must be positive-height/width RGB, "
            f"got {(height, width, channels)}"
        )
    actual_aspect_ratio = width / height
    relative_error = abs(actual_aspect_ratio / config.expected_input_aspect_ratio - 1.0)
    if relative_error > config.aspect_ratio_relative_tolerance:
        raise ValueError(
            "cam_high input aspect ratio does not match the ROI contract: "
            f"actual={actual_aspect_ratio:.9g} "
            f"expected={config.expected_input_aspect_ratio:.9g} "
            f"relative_error={relative_error:.6g} "
            f"tolerance={config.aspect_ratio_relative_tolerance:.6g}"
        )

    left_n, top_n, right_n, bottom_n = config.normalized_ltrb
    left = int(math.floor(width * left_n + 1e-9))
    top = int(math.floor(height * top_n + 1e-9))
    right = int(math.ceil(width * right_n - 1e-9))
    bottom = int(math.ceil(height * bottom_n - 1e-9))
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            "normalized cam_high ROI resolved to invalid pixel bounds: "
            f"{(left, top, right, bottom)} for {(height, width, channels)}"
        )
    return CamHighRoiBounds(
        input_height=height,
        input_width=width,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )


def crop_cam_high_roi(
    image: np.ndarray,
    *,
    config: CamHighRoiConfig = TASK_V1_CAM_HIGH_ROI,
) -> CamHighRoiResult:
    """Convert and crop one ``cam_high`` frame using the validated contract."""

    hwc, source_layout, source_dtype = to_hwc_uint8(image)
    bounds = resolve_cam_high_roi_bounds(hwc.shape, config=config)
    row_slice, column_slice = bounds.slices
    cropped = np.ascontiguousarray(hwc[row_slice, column_slice, :])
    return CamHighRoiResult(
        image=cropped,
        bounds=bounds,
        config=config,
        source_layout=source_layout,
        source_dtype=source_dtype,
    )


__all__ = [
    "CamHighRoiBounds",
    "CamHighRoiConfig",
    "CamHighRoiResult",
    "TASK_V1_CAM_HIGH_ASPECT_RATIO_REL_TOLERANCE",
    "TASK_V1_CAM_HIGH_INPUT_ASPECT_RATIO",
    "TASK_V1_CAM_HIGH_NORMALIZED_LTRB",
    "TASK_V1_CAM_HIGH_ROI",
    "crop_cam_high_roi",
    "resolve_cam_high_roi_bounds",
    "to_hwc_uint8",
]
