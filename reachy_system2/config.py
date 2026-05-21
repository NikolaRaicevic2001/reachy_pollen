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
