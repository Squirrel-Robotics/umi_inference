"""Robot-side state construction and strict shared-anchor SE(3) decoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


V2_TO_CURRENT_ACTION_BASIS_SCHEMA = "umi_v2_to_current_action_basis_v1"


def _pose(pose: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite (4,4) transform")
    return pose


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([rotation[..., :3, 0], rotation[..., :3, 1]], axis=-1)


def rot6d_to_matrix(rot6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Gram-Schmidt decode of column-major [R[:,0], R[:,1]]."""

    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6 or not np.all(np.isfinite(rot6d)):
        raise ValueError("rot6d must be finite with last dimension 6")
    c0, c1 = rot6d[..., :3], rot6d[..., 3:]
    n0 = np.linalg.norm(c0, axis=-1, keepdims=True)
    if np.any(n0 < eps):
        raise ValueError("degenerate first rot6d column")
    b0 = c0 / n0
    c1 = c1 - np.sum(b0 * c1, axis=-1, keepdims=True) * b0
    n1 = np.linalg.norm(c1, axis=-1, keepdims=True)
    if np.any(n1 < eps):
        raise ValueError("degenerate second rot6d column")
    b1 = c1 / n1
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)


def _project_near_rotation(matrix: np.ndarray, name: str) -> np.ndarray:
    """Validate a rounded near-SO(3) calibration and project it onto SO(3)."""

    raw = np.asarray(matrix, dtype=np.float64)
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must be a finite (3,3) matrix")
    determinant = float(np.linalg.det(raw))
    orthogonality_error = float(np.linalg.norm(raw.T @ raw - np.eye(3)))
    if determinant <= 0.0 or abs(determinant - 1.0) > 1e-3:
        raise ValueError(f"{name} determinant is not near +1: {determinant}")
    if orthogonality_error > 1e-3:
        raise ValueError(
            f"{name} is too far from SO(3): error={orthogonality_error}"
        )
    u, _, vt = np.linalg.svd(raw)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def load_v2_to_current_action_bases(
    path: str | Path,
) -> dict[str, np.ndarray]:
    """Load ``A = R_v2.T @ R_current`` for both robot arms."""

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != V2_TO_CURRENT_ACTION_BASIS_SCHEMA:
        raise ValueError(
            f"{config_path}: unexpected schema {payload.get('schema')!r}"
        )
    if payload.get("direction") != "v2_to_current":
        raise ValueError(
            f"{config_path}: direction must be 'v2_to_current'"
        )
    raw_bases = payload.get("R_v2_from_current")
    if not isinstance(raw_bases, dict):
        raise ValueError(f"{config_path}: R_v2_from_current must be a mapping")
    return {
        side: _project_near_rotation(
            raw_bases.get(side), f"{config_path}:R_v2_from_current:{side}"
        )
        for side in ("left", "right")
    }


def convert_v2_actions_to_current(
    actions: np.ndarray,
    v2_from_current: dict[str, np.ndarray],
) -> np.ndarray:
    """Rotate arm delta actions from the trained v2 basis to current axes.

    For each side, ``A`` maps current-basis vector components to v2-basis
    components. Therefore model output is converted with
    ``p_current = A.T @ p_v2`` and
    ``dR_current = A.T @ dR_v2 @ A``. Absolute hand targets are unchanged.
    """

    source = np.asarray(actions, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 30 or not np.all(np.isfinite(source)):
        raise ValueError(f"actions must be finite (H,30), got {source.shape}")
    result = source.copy()
    for side, start in (("left", 0), ("right", 9)):
        if side not in v2_from_current:
            raise ValueError(f"v2_from_current is missing {side}")
        basis = _project_near_rotation(
            v2_from_current[side], f"{side}_R_v2_from_current"
        )
        result[:, start : start + 3] = np.einsum(
            "ji,hj->hi", basis, source[:, start : start + 3]
        )
        rotation_v2 = rot6d_to_matrix(source[:, start + 3 : start + 9])
        rotation_current = np.einsum(
            "ji,hjk,kl->hil", basis, rotation_v2, basis
        )
        result[:, start + 3 : start + 9] = matrix_to_rot6d(rotation_current)
    return result


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert one SO(3) matrix to the normalized ROS/SDK [x,y,z,w] order."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite (3,3) matrix")
    # Project tiny numerical drift back onto SO(3) before conversion.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
            w = (rotation[2, 1] - rotation[1, 2]) / scale
        elif index == 1:
            scale = np.sqrt(max(0.0, 1.0 - rotation[0, 0] + rotation[1, 1] - rotation[2, 2])) * 2.0
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
            w = (rotation[0, 2] - rotation[2, 0]) / scale
        else:
            scale = np.sqrt(max(0.0, 1.0 - rotation[0, 0] - rotation[1, 1] + rotation[2, 2])) * 2.0
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
            w = (rotation[1, 0] - rotation[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("rotation produced an invalid quaternion")
    quaternion /= norm
    # q and -q encode the same rotation. Canonicalize for stable command logs.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def relative_pose(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_pose(previous, "previous")) @ _pose(current, "current")


def _change_of_basis(transform: np.ndarray, name: str) -> np.ndarray:
    """Validate a fixed SE(3) change of basis used by state/action mapping."""

    transform = _pose(transform, name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} must have homogeneous bottom row [0,0,0,1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform


def load_controller_to_ee_transforms(path: str | Path) -> dict[str, np.ndarray]:
    """Load the same controller-to-EEF bases used by ``pred_to_robot``.

    The current config stores rotations only, so the translation is zero.  A
    later calibration may provide ``T_left``/``T_right`` full SE(3) matrices;
    those take precedence without changing the live decoder.
    """

    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        transform_key = f"T_{side}"
        rotation_key = f"R_{side}"
        if transform_key in payload:
            transform = np.asarray(payload[transform_key], dtype=np.float64)
        elif rotation_key in payload:
            rotation = np.asarray(payload[rotation_key], dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError(
                    f"{config_path}: {rotation_key} must have shape (3,3)"
                )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
        else:
            raise ValueError(
                f"{config_path}: missing {transform_key} or {rotation_key}"
            )
        result[side] = _change_of_basis(
            transform, f"{config_path}:{side}_controller_to_ee"
        )
    return result


def build_state30(
    previous_left: np.ndarray,
    current_left: np.ndarray,
    previous_right: np.ndarray,
    current_right: np.ndarray,
    left_hand_deg: np.ndarray,
    right_hand_deg: np.ndarray,
    *,
    controller_to_ee: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build the 10 Hz adjacent controller-frame delta state used by training.

    Robot feedback is expressed in the physical EEF basis.  When a fixed
    controller-to-EEF mapping ``B`` is supplied, invert the action mapping:
    ``delta_controller = inv(B) @ delta_ee @ B``.
    """

    blocks = []
    bases = controller_to_ee or {
        "left": np.eye(4, dtype=np.float64),
        "right": np.eye(4, dtype=np.float64),
    }
    for side, previous, current in (
        ("left", previous_left, current_left),
        ("right", previous_right, current_right),
    ):
        delta_ee = relative_pose(previous, current)
        basis = _change_of_basis(bases[side], f"{side}_controller_to_ee")
        delta_controller = np.linalg.inv(basis) @ delta_ee @ basis
        blocks.append(
            np.concatenate(
                [
                    delta_controller[:3, 3],
                    matrix_to_rot6d(delta_controller[:3, :3]),
                ]
            )
        )
    for hand, name in ((left_hand_deg, "left_hand_deg"), (right_hand_deg, "right_hand_deg")):
        hand = np.asarray(hand, dtype=np.float64)
        if hand.shape != (6,) or not np.all(np.isfinite(hand)):
            raise ValueError(f"{name} must be finite with shape (6,)")
        blocks.append(hand)
    return np.concatenate(blocks).astype(np.float32)


def decode_shared_anchor_actions(
    actions: np.ndarray,
    current_left: np.ndarray,
    current_right: np.ndarray,
    *,
    composition: str = "right",
) -> dict[str, np.ndarray]:
    """Decode action (H,30) into shared-anchor absolute targets.

    ``right`` computes ``anchor @ delta`` (body/tool-frame delta).
    ``left`` computes ``delta @ anchor`` (base/world-frame delta).  Full SE(3)
    left multiplication also rotates the anchor position about the base origin.
    """

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 30 or not np.all(np.isfinite(actions)):
        raise ValueError(f"actions must be finite (H,30), got {actions.shape}")
    if composition not in ("right", "left"):
        raise ValueError(f"composition must be 'right' or 'left', got {composition!r}")
    anchors = (_pose(current_left, "current_left"), _pose(current_right, "current_right"))
    absolute_targets = []
    for start, anchor in zip((0, 9), anchors, strict=True):
        delta = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
        delta[:, :3, 3] = actions[:, start : start + 3]
        delta[:, :3, :3] = rot6d_to_matrix(actions[:, start + 3 : start + 9])
        # Every horizon target uses the same observation-time anchor.  The
        # selectable left mode is experimental and intentionally changes the
        # frame convention; it never cumulatively chains horizon targets.
        if composition == "right":
            absolute_targets.append(anchor[None, ...] @ delta)
        else:
            absolute_targets.append(delta @ anchor[None, ...])
    return {
        "left_target_T": absolute_targets[0],
        "right_target_T": absolute_targets[1],
        "left_hand_deg": actions[:, 18:24].astype(np.float32),
        "right_hand_deg": actions[:, 24:30].astype(np.float32),
    }


def decoded_targets_to_pose7(decoded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert absolute target transforms to SDK pose7 [x,y,z,qx,qy,qz,qw]."""

    output: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        key = f"{side}_target_T"
        transforms = np.asarray(decoded[key], dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"{key} must have shape (H,4,4), got {transforms.shape}")
        pose7 = np.empty((len(transforms), 7), dtype=np.float64)
        pose7[:, :3] = transforms[:, :3, 3]
        for index, transform in enumerate(transforms):
            pose7[index, 3:] = matrix_to_quaternion_xyzw(transform[:3, :3])
        output[f"{side}_target_pose7"] = pose7
    output["left_hand_deg"] = np.asarray(decoded["left_hand_deg"], dtype=np.float32)
    output["right_hand_deg"] = np.asarray(decoded["right_hand_deg"], dtype=np.float32)
    return output


def decode_proxy_actions_for_tcp(
    actions: np.ndarray,
    current_left_tcp: np.ndarray,
    current_right_tcp: np.ndarray,
    left_tcp_to_proxy: np.ndarray,
    right_tcp_to_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decode model-proxy deltas into physical TCP targets using calibrated extrinsics.

    ``tcp_to_proxy`` is defined by ``T_world_proxy = T_world_tcp @ tcp_to_proxy``.
    Therefore a model body delta is conjugated as
    ``delta_tcp = tcp_to_proxy @ delta_proxy @ inv(tcp_to_proxy)``.
    The extrinsics are required; silently assuming identity is unsafe.
    """

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 30 or not np.all(np.isfinite(actions)):
        raise ValueError(f"actions must be finite (H,30), got {actions.shape}")
    tcp_anchors = (
        _pose(current_left_tcp, "current_left_tcp"),
        _pose(current_right_tcp, "current_right_tcp"),
    )
    extrinsics = (
        _pose(left_tcp_to_proxy, "left_tcp_to_proxy"),
        _pose(right_tcp_to_proxy, "right_tcp_to_proxy"),
    )
    targets = []
    for start, anchor, tcp_to_proxy in zip((0, 9), tcp_anchors, extrinsics, strict=True):
        delta_proxy = np.broadcast_to(np.eye(4), (len(actions), 4, 4)).copy()
        delta_proxy[:, :3, 3] = actions[:, start : start + 3]
        delta_proxy[:, :3, :3] = rot6d_to_matrix(actions[:, start + 3 : start + 9])
        proxy_to_tcp = np.linalg.inv(tcp_to_proxy)
        delta_tcp = tcp_to_proxy[None, ...] @ delta_proxy @ proxy_to_tcp[None, ...]
        targets.append(anchor[None, ...] @ delta_tcp)
    return {
        "left_target_T": targets[0],
        "right_target_T": targets[1],
        "left_hand_deg": actions[:, 18:24].astype(np.float32),
        "right_hand_deg": actions[:, 24:30].astype(np.float32),
    }


def decode_controller_actions_for_ee(
    actions: np.ndarray,
    current_left_ee: np.ndarray,
    current_right_ee: np.ndarray,
    controller_to_ee: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply the ``pred_to_robot`` mapping, then right-compose one shared anchor.

    For each side and horizon step this computes exactly
    ``delta_ee = B @ delta_controller @ inv(B)`` followed by
    ``target_ee = current_ee @ delta_ee``.  Horizon targets are independent;
    they are never chained together.
    """

    left = _change_of_basis(controller_to_ee["left"], "left_controller_to_ee")
    right = _change_of_basis(controller_to_ee["right"], "right_controller_to_ee")
    return decode_proxy_actions_for_tcp(
        actions,
        current_left_ee,
        current_right_ee,
        left,
        right,
    )


def assert_safe_first_step(
    decoded: dict[str, np.ndarray],
    current_left: np.ndarray,
    current_right: np.ndarray,
    *,
    max_translation_m: float = 0.05,
    max_rotation_rad: float = 0.35,
) -> None:
    """Minimal gate before a downstream controller accepts the first target."""

    for key, current in (("left_target_T", current_left), ("right_target_T", current_right)):
        delta = np.linalg.inv(_pose(current, key)) @ decoded[key][0]
        translation = float(np.linalg.norm(delta[:3, 3]))
        cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        rotation = float(np.arccos(cosine))
        if translation > max_translation_m or rotation > max_rotation_rad:
            raise RuntimeError(
                f"unsafe {key}: translation={translation:.4f}m, rotation={rotation:.4f}rad"
            )
