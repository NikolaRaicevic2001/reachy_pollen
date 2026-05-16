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
            xmin=_f("SAFE_WORKSPACE_XMIN", 0.1),
            xmax=_f("SAFE_WORKSPACE_XMAX", 0.8),
            ymin=_f("SAFE_WORKSPACE_YMIN", -0.6),
            ymax=_f("SAFE_WORKSPACE_YMAX", 0.6),
            zmin=_f("SAFE_WORKSPACE_ZMIN", -0.5),
            zmax=_f("SAFE_WORKSPACE_ZMAX", 0.5),
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


def require_every_tracked_label_default() -> bool:
    """If true, planning waits until each SYSTEM2_LABELS entry matches some detection."""
    raw = (os.environ.get("SYSTEM2_REQUIRE_ALL_LABELS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
