"""Pollen Vision Perception + SDK camera wrapper for System 2 scene context."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np

from reachy2_sdk.utils.utils import invert_affine_transformation_matrix

from reachy_system2.config import (
    perception_detection_threshold_default,
    perception_freq_default,
    settling_s_default,
)

logger = logging.getLogger(__name__)


def to_robot_frame(T_reachy_cam: np.ndarray, pose_cam: np.ndarray) -> np.ndarray:
    """Apply camera→robot transform to a 4×4 pose in the camera frame.

    Pollen `Perception(r_cam, T_reachy_cam, ...)` typically already returns poses
    in the robot frame; use this only for raw camera-frame poses.
    """
    return T_reachy_cam @ pose_cam


def _translation_robot(pose_4x4: np.ndarray, T_reachy_cam: np.ndarray | None, assume_robot_frame: bool) -> np.ndarray:
    """Return 3-vector position in robot base frame."""
    p = np.asarray(pose_4x4, dtype=float)
    if p.shape != (4, 4):
        raise ValueError(f"Expected 4×4 pose, got shape {p.shape}")
    if assume_robot_frame or T_reachy_cam is None:
        return p[:3, 3].copy()
    return to_robot_frame(T_reachy_cam, p)[:3, 3].copy()


def format_scene_for_llm(
    detected_objects: Sequence[dict[str, Any]],
    *,
    T_reachy_cam: np.ndarray | None = None,
    assume_poses_already_in_robot_frame: bool = True,
) -> str:
    """Build a stable text block for the LLM (robot-frame positions + scores)."""
    lines: list[str] = ["DETECTED_OBJECTS (robot frame, meters; +Y = robot's left):", ""]
    if not detected_objects:
        lines.append("(none above threshold)")
        return "\n".join(lines)

    for i, obj in enumerate(detected_objects):
        name = obj.get("name", f"object_{i}")
        pose = obj.get("pose")
        if pose is None:
            continue
        t = _translation_robot(pose, T_reachy_cam, assume_poses_already_in_robot_frame)
        det = obj.get("detection_score", obj.get("score"))
        temp = obj.get("temporal_score")
        parts = [f"- {name}: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}"]
        if det is not None:
            parts.append(f"detection_score={float(det):.3f}")
        if temp is not None:
            parts.append(f"temporal_score={float(temp):.3f}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


class System2Perception:
    """Owns PollenSDKCameraWrapper + pollen_vision Perception lifecycle."""

    def __init__(
        self,
        reachy: Any,
        *,
        freq: float | None = None,
        detection_threshold: float | None = None,
        assume_poses_already_in_robot_frame: bool = True,
    ) -> None:
        from pollen_vision.camera_wrappers.pollen_sdk_camera.pollen_sdk_camera_wrapper import (
            PollenSDKCameraWrapper,
        )
        from pollen_vision.perception import Perception

        self._reachy = reachy
        self._assume_robot = assume_poses_already_in_robot_frame
        self._freq = float(freq if freq is not None else perception_freq_default())
        self._default_detection_threshold = float(
            detection_threshold if detection_threshold is not None else perception_detection_threshold_default()
        )

        if reachy.cameras is None or reachy.cameras.depth is None:
            raise RuntimeError("Depth camera is required for Perception extrinsics (reachy.cameras.depth is None).")

        self._r_cam = PollenSDKCameraWrapper(reachy)
        T_cam_reachy = reachy.cameras.depth.get_extrinsics()
        self._T_reachy_cam = invert_affine_transformation_matrix(T_cam_reachy)

        self._perception = Perception(
            self._r_cam,
            self._T_reachy_cam,
            freq=self._freq,
            detection_threshold=self._default_detection_threshold,
        )
        self._started = False

    @property
    def T_reachy_cam(self) -> np.ndarray:
        return self._T_reachy_cam

    def set_tracked_labels(self, labels: Sequence[str]) -> None:
        self._perception.set_tracked_objects(list(labels))

    def start(self, *, visualize: bool = False) -> None:
        if self._started:
            return
        self._perception.start(visualize=visualize)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        stop_fn = getattr(self._perception, "stop", None)
        if callable(stop_fn):
            stop_fn()
        close_fn = getattr(self._perception, "close", None)
        if callable(close_fn):
            close_fn()
        self._started = False

    def snapshot(
        self,
        *,
        detection_threshold: float | None = None,
        settling_s: float | None = None,
    ) -> tuple[np.ndarray, str]:
        """Wait for mechanical/vision settling, then grab RGB + object list."""
        settle = float(settling_s if settling_s is not None else settling_s_default())
        if settle > 0:
            time.sleep(settle)

        data, _, _ = self._r_cam.get_data()
        if isinstance(data, dict):
            img = data.get("left")
            if img is None:
                img = data.get("rgb")
        else:
            img = data
        if img is None:
            raise RuntimeError("Camera get_data() returned no RGB image (expected 'left' or 'rgb' key).")

        thr = float(detection_threshold if detection_threshold is not None else self._default_detection_threshold)
        detected = self._perception.get_objects_infos(thr)
        scene = format_scene_for_llm(
            detected,
            T_reachy_cam=self._T_reachy_cam,
            assume_poses_already_in_robot_frame=self._assume_robot,
        )
        return np.asarray(img), scene
