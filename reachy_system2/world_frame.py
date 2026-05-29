"""World-frame scene memory using mobile-base odometry."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reachy_system2.perception import label_matches_tracked

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OdomSnapshot:
    x: float
    y: float
    theta_deg: float

    @classmethod
    def from_reachy(cls, reachy: Any) -> OdomSnapshot:
        mb = getattr(reachy, "mobile_base", None)
        if mb is None:
            return cls(0.0, 0.0, 0.0)
        o = mb.odometry
        return cls(float(o["x"]), float(o["y"]), float(o["theta"]))

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta_deg}


def reset_mobile_base_odometry(reachy: Any) -> None:
    """Set odometry origin to current base pose (world frame for this run)."""
    mb = getattr(reachy, "mobile_base", None)
    if mb is None:
        return
    reset = getattr(mb, "reset_odometry", None)
    if callable(reset):
        reset()
        logger.info("Mobile base odometry reset (world origin).")


def xyz_from_object_pose(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    pose = obj.get("pose")
    if pose is None:
        return None
    p = np.asarray(pose, dtype=float)
    if p.shape != (4, 4):
        return None
    t = p[:3, 3]
    return float(t[0]), float(t[1]), float(t[2])


def robot_to_world(
    xyz_robot: tuple[float, float, float],
    odom_at_observation: OdomSnapshot,
) -> tuple[float, float, float]:
    """Robot-frame point at observation time → fixed odometry/world frame."""
    x, y, z = xyz_robot
    th = math.radians(odom_at_observation.theta_deg)
    bx, by = odom_at_observation.x, odom_at_observation.y
    c, s = math.cos(th), math.sin(th)
    xw = bx + c * x - s * y
    yw = by + s * x + c * y
    return xw, yw, z


def world_to_robot(
    xyz_world: tuple[float, float, float],
    odom_now: OdomSnapshot,
) -> tuple[float, float, float]:
    """World/odometry point → current robot base frame."""
    xw, yw, z = xyz_world
    th = math.radians(odom_now.theta_deg)
    bx, by = odom_now.x, odom_now.y
    dx, dy = xw - bx, yw - by
    c, s = math.cos(th), math.sin(th)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    return xr, yr, z


@dataclass
class WorldObject:
    name: str
    x: float
    y: float
    z: float


@dataclass
class SceneMemory:
    """Persistent map of detections in odometry/world frame."""

    objects: dict[str, WorldObject] = field(default_factory=dict)

    def update_from_detections(
        self,
        detected_objects: Sequence[dict[str, Any]],
        odom: OdomSnapshot,
    ) -> None:
        for obj in detected_objects:
            xyz = xyz_from_object_pose(obj)
            if xyz is None:
                continue
            name = str(obj.get("name", "")).strip()
            if not name:
                continue
            xw, yw, wz = robot_to_world(xyz, odom)
            self.objects[name] = WorldObject(name=name, x=xw, y=yw, z=wz)

    def robot_xyz_for_label(
        self,
        label: str,
        odom: OdomSnapshot,
    ) -> tuple[float, float, float] | None:
        for name, wo in self.objects.items():
            if label_matches_tracked(label, name):
                return world_to_robot((wo.x, wo.y, wo.z), odom)
        return None

    def format_for_llm(self, odom: OdomSnapshot) -> str:
        lines = [
            "WORLD_MEMORY (odometry/world map; xyz below converted to CURRENT robot frame):",
            f"MOBILE_BASE_ODOMETRY: x={odom.x:.3f}, y={odom.y:.3f}, theta_deg={odom.theta_deg:.1f}",
            "",
        ]
        if not self.objects:
            lines.append("(no objects stored yet)")
            return "\n".join(lines)
        for wo in sorted(self.objects.values(), key=lambda o: o.name):
            rx, ry, rz = world_to_robot((wo.x, wo.y, wo.z), odom)
            lines.append(f"- {wo.name}: x={rx:.4f}, y={ry:.4f}, z={rz:.4f}")
        lines.append(
            "\nNote: world positions stay fixed when the base moves; use mobile_base_translate_by / "
            "rotate_by to bring distant targets into arm reach, then use CURRENT robot-frame xyz for goto_pose."
        )
        return "\n".join(lines)

    def find_robot_xyz(
        self,
        *,
        labels: Sequence[str],
        detected_objects: Sequence[dict[str, Any]],
        odom: OdomSnapshot,
    ) -> tuple[str, tuple[float, float, float]] | None:
        """Prefer live detection; fall back to world memory."""
        for obj in detected_objects:
            name = str(obj.get("name", ""))
            if not any(label_matches_tracked(lab, name) for lab in labels):
                continue
            xyz = xyz_from_object_pose(obj)
            if xyz is not None:
                return name, xyz
        for lab in labels:
            mem = self.robot_xyz_for_label(lab, odom)
            if mem is not None:
                return lab, mem
        return None
