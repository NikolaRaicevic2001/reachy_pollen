"""Pollen Vision Perception + SDK camera wrapper for System 2 scene context."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from reachy2_sdk.utils.utils import invert_affine_transformation_matrix

from reachy_system2.config import (
    perception_detection_threshold_default,
    perception_freq_default,
    perception_retry_settling_s_default,
    perception_snapshot_max_attempts_default,
    settling_s_default,
)

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """One perception sample: RGB frame, annotated frame, LLM scene text, structured detections."""

    rgb: np.ndarray
    annotated_rgb: np.ndarray
    scene: str
    objects: list[dict[str, Any]]
    depth_viz_rgb: np.ndarray | None = None


def to_robot_frame(T_reachy_cam: np.ndarray, pose_cam: np.ndarray) -> np.ndarray:
    """Apply camera→robot transform to a 4×4 pose in the camera frame.

    Pollen `Perception(r_cam, T_reachy_cam, ...)` typically already returns poses
    in the robot frame; use this only for raw camera-frame poses.
    """
    return T_reachy_cam @ pose_cam


def _translation_robot(
    pose_4x4: np.ndarray,
    T_reachy_cam: np.ndarray | None,
    assume_robot_frame: bool,
) -> np.ndarray:
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


def _norm_label(s: str) -> str:
    return str(s).strip().lower()


def label_matches_tracked(label: str, object_name: str) -> bool:
    """Loose match: OWL-ViT names and user labels often differ slightly."""
    a, b = _norm_label(label), _norm_label(object_name)
    if not a or not b:
        return False
    return a in b or b in a


def _object_names(objects: Sequence[dict[str, Any]]) -> list[str]:
    return [str(o.get("name", "")) for o in objects]


def _unique_detection_names(objects: Sequence[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for n in _object_names(objects):
        if n and n not in seen:
            seen.append(n)
    return seen


def missing_tracked_labels(
    objects: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> list[str]:
    """Labels with no loose match to any filtered detection name."""
    names = _object_names(objects)
    return [lab for lab in labels if not any(label_matches_tracked(lab, n) for n in names)]


def objects_satisfy_tracked_labels(
    objects: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> bool:
    """Whether each tracked label matches at least one detected object name (not object count)."""
    if not objects:
        return False
    if not labels:
        return True
    return not missing_tracked_labels(objects, labels)


def _detection_predictions_from_objects(
    detected_objects: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert ``get_objects_infos`` dicts to OWL-ViT-style predictions for ``Annotator``."""
    predictions: list[dict[str, Any]] = []
    for obj in detected_objects:
        bbox = obj.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = (int(v) for v in bbox)
        predictions.append(
            {
                "label": str(obj.get("name", "")),
                "score": float(obj.get("detection_score", obj.get("score", 0.0))),
                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
        )
    return predictions


def depth_to_viz_rgb(
    depth: np.ndarray,
    *,
    target_hw: tuple[int, int] | None = None,
    max_range_m: float = 2.0,
) -> np.ndarray:
    """Encode torso depth as a pseudo-RGB colormap for vision LLMs (same viewpoint as RGB)."""
    import cv2

    d = np.asarray(depth, dtype=np.float32).squeeze()
    if d.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {d.shape}")

    valid = d > 0
    if not np.any(valid):
        out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    else:
        d_m = d.copy()
        if float(np.nanmax(d_m[valid])) > 50.0:
            d_m = d_m / 1000.0
        d_valid = d_m[valid]
        near = float(np.percentile(d_valid, 5))
        far = float(np.percentile(d_valid, 95))
        span = max(far - near, 1e-3)
        far = min(far, near + max_range_m)
        norm = np.zeros(d.shape, dtype=np.uint8)
        t = np.clip((d_m[valid] - near) / span, 0.0, 1.0)
        norm[valid] = (255.0 * (1.0 - t)).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        colored_bgr[~valid] = 0
        out = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    if target_hw is not None and (out.shape[0], out.shape[1]) != target_hw:
        out = cv2.resize(out, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return out


def annotate_rgb(rgb: np.ndarray, detected_objects: Sequence[dict[str, Any]]) -> np.ndarray:
    """Draw boxes (and masks when present) for filtered Perception objects on ``rgb``."""
    from pollen_vision.utils import Annotator

    im = np.asarray(rgb)
    if im.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {im.shape}")

    predictions = _detection_predictions_from_objects(detected_objects)
    if not predictions:
        return im.copy()

    masks: list[np.ndarray] = []
    for obj in detected_objects:
        bbox = obj.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        mask = obj.get("mask")
        if mask is not None:
            masks.append(np.asarray(mask))

    annotator = Annotator()
    if len(masks) == len(predictions):
        return annotator.annotate(im=im, detection_predictions=predictions, masks=masks)
    return annotator.annotate(im=im, detection_predictions=predictions)


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
    ) -> PerceptionSnapshot:
        """Wait for settling, grab RGB, filtered detections, annotated frame, and LLM scene text."""
        settle = float(settling_s if settling_s is not None else settling_s_default())
        if settle > 0:
            time.sleep(settle)

        data, _, _ = self._r_cam.get_data()
        depth_raw = None
        if isinstance(data, dict):
            img = data.get("left")
            if img is None:
                img = data.get("rgb")
            depth_raw = data.get("depth")
        else:
            img = data
        if img is None:
            raise RuntimeError("Camera get_data() returned no RGB image (expected 'left' or 'rgb' key).")

        rgb = np.asarray(img)
        depth_viz_rgb = None
        if depth_raw is not None:
            try:
                depth_viz_rgb = depth_to_viz_rgb(
                    depth_raw,
                    target_hw=(rgb.shape[0], rgb.shape[1]),
                )
            except Exception as exc:
                logger.warning("Depth colormap failed (LLM will get RGB only): %s", exc)

        thr = float(detection_threshold if detection_threshold is not None else self._default_detection_threshold)
        detected = self._perception.get_objects_infos(thr)
        scene = format_scene_for_llm(
            detected,
            T_reachy_cam=self._T_reachy_cam,
            assume_poses_already_in_robot_frame=self._assume_robot,
        )
        annotated_rgb = annotate_rgb(rgb, detected)
        return PerceptionSnapshot(
            rgb=rgb,
            annotated_rgb=annotated_rgb,
            scene=scene,
            objects=list(detected),
            depth_viz_rgb=depth_viz_rgb,
        )

    def snapshot_until_tracked_objects(
        self,
        *,
        labels: Sequence[str],
        detection_threshold: float | None = None,
        settling_s: float | None = None,
        max_attempts: int | None = None,
        retry_settling_s: float | None = None,
    ) -> PerceptionSnapshot:
        """Repeatedly grab frames until detections match ``labels``, or raise after ``max_attempts``.

        Pollen's object filter often needs several ticks after ``start()`` before poses appear.
        """
        max_att = int(max_attempts if max_attempts is not None else perception_snapshot_max_attempts_default())
        base_settle = float(settling_s if settling_s is not None else settling_s_default())
        retry_settle = float(retry_settling_s if retry_settling_s is not None else perception_retry_settling_s_default())

        last: PerceptionSnapshot | None = None
        for attempt in range(1, max_att + 1):
            settle = base_settle if attempt == 1 else retry_settle
            snap = self.snapshot(detection_threshold=detection_threshold, settling_s=settle)
            last = snap
            if objects_satisfy_tracked_labels(snap.objects, labels):
                if attempt > 1:
                    logger.info("Perception criteria met after %s attempts (%s objects).", attempt, len(snap.objects))
                return snap
            missing = missing_tracked_labels(snap.objects, labels)
            logger.warning(
                "Perception attempt %s/%s: %s filtered objects, detection names=%s, "
                "tracked labels=%s, missing label match=%s",
                attempt,
                max_att,
                len(snap.objects),
                _unique_detection_names(snap.objects),
                list(labels),
                missing,
            )

        assert last is not None
        missing = missing_tracked_labels(last.objects, labels)
        raise RuntimeError(
            f"Perception did not satisfy criteria after {max_att} attempts: "
            f"each tracked label must match a detection name (loose substring), not merely N objects. "
            f"Last snapshot: {len(last.objects)} object(s), names={_unique_detection_names(last.objects)!r}, "
            f"labels={list(labels)!r}, no match for {missing!r}. "
            "Try lowering PERCEPTION_DETECTION_THRESHOLD, renaming labels to match OWL-ViT output, "
            "or raising SYSTEM2_SNAPSHOT_MAX_ATTEMPTS."
        )
