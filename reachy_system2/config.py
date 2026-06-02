"""Load numeric / string defaults from environment (see repo-root `.env.example`)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _i(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(float(raw))


@dataclass(frozen=True)
class SafeWorkspace:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def from_env(cls) -> SafeWorkspace:
        return cls(
            # +X forward: allow negative xmin so the arm can retract backward (away from table).
            xmin=_f("SAFE_WORKSPACE_XMIN", -0.20),
            xmax=_f("SAFE_WORKSPACE_XMAX", 0.85),
            ymin=_f("SAFE_WORKSPACE_YMIN", -0.65),
            ymax=_f("SAFE_WORKSPACE_YMAX", 0.65),
            zmin=_f("SAFE_WORKSPACE_ZMIN", -0.55),
            zmax=_f("SAFE_WORKSPACE_ZMAX", 0.55),
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )


def settling_s_default() -> float:
    return _f("SETTLING_S", 0.75)


def perception_freq_default() -> float:
    return _f("PERCEPTION_FREQ", 40.0)


def perception_detection_threshold_default() -> float:
    return _f("PERCEPTION_DETECTION_THRESHOLD", 0.1)

# add alternative perception model YOLO
def perception_detector_default() -> str:
    return os.environ.get("PERCEPTION_DETECTOR", "owl_vit").strip().lower()


def yolo_world_model_default() -> str:
    return os.environ.get("YOLO_WORLD_MODEL", "yolov8s-worldv2.pt").strip()


def yolo_world_device_default() -> str | None:
    raw = os.environ.get("YOLO_WORLD_DEVICE")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()

def perception_snapshot_max_attempts_default() -> int:
    """How many `snapshot()` tries before planning when waiting for detections."""
    return _i("SYSTEM2_SNAPSHOT_MAX_ATTEMPTS", 20)


def perception_retry_settling_s_default() -> float:
    """Sleep before each retry after the first snapshot (seconds)."""
    return _f("PERCEPTION_RETRY_SETTLING_S", 0.5)


def clearance_margin_m_default() -> float:
    """Meters above highest support surface z for transit waypoints."""
    return _f("SYSTEM2_CLEARANCE_MARGIN_M", 0.15)


def min_lift_above_tcp_m_default() -> float:
    """Minimum vertical rise from current TCP z when computing z_safe."""
    return _f("SYSTEM2_MIN_LIFT_ABOVE_TCP_M", 0.12)


def link_clearance_m_default() -> float:
    """Extra z above support+margin for arm links hanging below the TCP."""
    return _f("SYSTEM2_LINK_CLEARANCE_M", 0.10)


def support_xy_radius_m_default() -> float:
    """Ignore support fixtures farther than this (xy) from manipulandum cluster."""
    return _f("SYSTEM2_SUPPORT_XY_RADIUS_M", 0.70)


def hover_above_object_m_default() -> float:
    """Minimum z clearance above highest manipulandum centroid (+Z up)."""
    return _f("SYSTEM2_HOVER_ABOVE_OBJECT_M", 0.12)


def verify_max_retries_default() -> int:
    """After FAILED verification, how many failure-replan attempts per planned subtask."""
    return _i("SYSTEM2_MAX_VERIFY_RETRIES", 2)


def mobile_base_timeout_s_default() -> float:
    """Default timeout for mobile base moves when wait=True (seconds)."""
    return _f("SYSTEM2_MOBILE_BASE_TIMEOUT_S", 10.0)


@dataclass(frozen=True)
class ArmReachBand:
    """Comfortable arm reach in robot frame (subset of SafeWorkspace). Outside → use mobile base."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    # Require pick xy x ≤ (xmax - forward_x_margin) so the arm is not at maximum stretch (+X).
    forward_x_margin: float

    @classmethod
    def from_env(cls) -> ArmReachBand:
        return cls(
            xmin=_f("ARM_REACH_XMIN", 0.22),
            xmax=_f("ARM_REACH_XMAX", 0.72),
            ymin=_f("ARM_REACH_YMIN", -0.55),
            ymax=_f("ARM_REACH_YMAX", 0.55),
            forward_x_margin=_f("ARM_REACH_FORWARD_MARGIN_M", 0.20),
        )

    def effective_xmax(self) -> float:
        return self.xmax - self.forward_x_margin

    def contains_xy(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.effective_xmax() and self.ymin <= y <= self.ymax


def max_base_approach_rounds_default() -> int:
    return _i("SYSTEM2_MAX_BASE_APPROACH_ROUNDS", 3)


def reset_odometry_on_run_default() -> bool:
    raw = os.environ.get("SYSTEM2_RESET_ODOMETRY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
