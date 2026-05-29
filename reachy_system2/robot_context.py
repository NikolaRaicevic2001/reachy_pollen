"""Summarize live robot state for LLM prompts."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Sequence

import numpy as np

from reachy_system2.config import (
    SafeWorkspace,
    clearance_margin_m_default,
    hover_above_object_m_default,
    link_clearance_m_default,
    min_lift_above_tcp_m_default,
    support_xy_radius_m_default,
)

logger = logging.getLogger(__name__)

_OBJECT_LINE = re.compile(
    r"^- (.+?): x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)",
    re.MULTILINE,
)
_SUPPORT_WORDS = ("table", "counter", "tray", "desk", "surface", "workbench", "shelf")


def parse_scene_objects(scene_description: str) -> list[tuple[str, float, float, float]]:
    out: list[tuple[str, float, float, float]] = []
    for m in _OBJECT_LINE.finditer(scene_description):
        out.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return out


def _is_support_fixture(name: str) -> bool:
    n = name.lower()
    return any(w in n for w in _SUPPORT_WORDS)


def _manipulandum_objects(
    objects: Sequence[tuple[str, float, float, float]],
) -> list[tuple[str, float, float, float]]:
    return [(n, x, y, z) for n, x, y, z in objects if not _is_support_fixture(n)]


def _robust_support_z(
    objects: Sequence[tuple[str, float, float, float]],
    *,
    xy_radius_m: float,
) -> float | None:
    """Support surface z (+Z up = top of surface is the largest z among valid support points)."""
    manip = _manipulandum_objects(objects)
    support = [(n, x, y, z) for n, x, y, z in objects if _is_support_fixture(n)]
    if not support:
        return None
    if manip:
        cx = sum(o[1] for o in manip) / len(manip)
        cy = sum(o[2] for o in manip) / len(manip)
        near = [
            z
            for _n, x, y, z in support
            if math.hypot(x - cx, y - cy) <= xy_radius_m
        ]
        if near:
            return max(near)
    # Fallback: median of support z (resists single far-outlier detections)
    zs = sorted(z for _n, _x, _y, z in support)
    return zs[len(zs) // 2]


def _estimate_table_z_below_objects(
    manip: Sequence[tuple[str, float, float, float]],
) -> float | None:
    """If support fixtures are missing/outliers, table is slightly below object centroids."""
    if not manip:
        return None
    return min(z for _n, _x, _y, z in manip) - 0.09


def _parse_tcps_from_robot_context(robot_context: str) -> dict[str, tuple[float, float, float]]:
    tcps: dict[str, tuple[float, float, float]] = {}
    for line in robot_context.splitlines():
        if "RIGHT_ARM" in line and "gripper TCP" in line:
            m = re.search(r"x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)", line)
            if m:
                tcps["r"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        if "LEFT_ARM" in line and "gripper TCP" in line:
            m = re.search(r"x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)", line)
            if m:
                tcps["l"] = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return tcps


def _tcp_xyz_from_fk(fk: Any) -> tuple[float, float, float]:
    m = np.asarray(fk, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix, got shape {m.shape}")
    t = m[:3, 3]
    return float(t[0]), float(t[1]), float(t[2])


def _rpy_deg_from_fk(fk: Any) -> tuple[float, float, float]:
    """Roll, pitch, yaw (degrees) — matches reachy2 get_pose_matrix (Rz @ Ry @ Rx)."""
    R = np.asarray(fk, dtype=float)[:3, :3]
    sy = math.sqrt(float(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy >= 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def format_workspace_bounds_for_llm(workspace: SafeWorkspace) -> str:
    """Allowed xyz box (executor rejects targets outside)."""
    return (
        "SAFE_WORKSPACE (robot base frame, meters; every goto_pose xyz must be inside):\n"
        f"x [{workspace.xmin:.3f}, {workspace.xmax:.3f}], "
        f"y [{workspace.ymin:.3f}, {workspace.ymax:.3f}], "
        f"z [{workspace.zmin:.3f}, {workspace.zmax:.3f}]"
    )


def format_end_effector_context_for_llm(reachy: Any) -> str:
    """Gripper TCP poses in robot base frame."""
    lines: list[str] = [
        "ROBOT_STATE (robot base frame, meters; +X forward, +Y robot left, +Z up):",
        "(rpy_deg below is FK readback; for top-down goto_pose use [0, 0, 0], not these values.)",
        "",
    ]
    any_arm = False
    for label, attr in (("RIGHT_ARM (r_arm)", "r_arm"), ("LEFT_ARM (l_arm)", "l_arm")):
        arm = getattr(reachy, attr, None)
        if arm is None:
            continue
        try:
            fk = arm.forward_kinematics()
            x, y, z = _tcp_xyz_from_fk(fk)
            roll, pitch, yaw = _rpy_deg_from_fk(fk)
            any_arm = True
            lines.append(
                f"- {label} gripper TCP: x={x:.4f}, y={y:.4f}, z={z:.4f}; "
                f"rpy_deg=[{roll:.1f}, {pitch:.1f}, {yaw:.1f}]"
            )
            gripper = getattr(arm, "gripper", None)
            if gripper is not None:
                opening = getattr(gripper, "present_position", None)
                if opening is None:
                    opening = getattr(gripper, "opening", None)
                if opening is not None:
                    lines.append(f"  gripper opening: {opening}")
        except Exception as e:
            logger.warning("%s forward_kinematics failed: %s", attr, e)
            lines.append(f"- {label}: pose unavailable ({e})")

    if not any_arm:
        lines.append("(no arm kinematics available)")
    return "\n".join(lines)


def format_scene_hints_for_llm(scene_description: str, robot_context: str) -> str:
    """Short numeric hints (support height, safe transit z) — not a scripted motion plan."""
    margin = clearance_margin_m_default()
    min_lift = min_lift_above_tcp_m_default()
    link_clr = link_clearance_m_default()
    hover_m = hover_above_object_m_default()
    xy_radius = support_xy_radius_m_default()

    objects = parse_scene_objects(scene_description)
    tcps = _parse_tcps_from_robot_context(robot_context)
    manip = _manipulandum_objects(objects)

    z_support = _robust_support_z(objects, xy_radius_m=xy_radius)
    if z_support is None:
        z_support = _estimate_table_z_below_objects(manip)

    z_candidates: list[float] = []
    for _arm, (_x, _y, tz) in tcps.items():
        z_candidates.append(tz + min_lift)
    if z_support is not None:
        z_candidates.append(z_support + margin + link_clr)
    if manip:
        z_obj_top = max(z for _n, _x, _y, z in manip)
        z_candidates.append(z_obj_top + hover_m)

    z_safe = max(z_candidates) if z_candidates else None

    lines = [
        "SCENE_HINTS (+Z is up: larger z = higher; hover above an object needs z > object z):",
    ]
    if z_support is not None:
        lines.append(f"- Estimated support surface z (table): {z_support:.3f}")
    pick: tuple[str, float, float, float] | None = None
    if manip:
        # Choose the tallest manipulandum as the primary pick target hint.
        pick = manip[0] if len(manip) == 1 else max(manip, key=lambda o: o[3])
    if manip:
        z_obj_top = max(z for _n, _x, _y, z in manip)
        lines.append(
            f"- Highest manipulandum z in PERCEPTION: {z_obj_top:.3f} "
            f"(hover z should be > {z_obj_top:.3f}, e.g. >= {z_obj_top + hover_m:.3f})"
        )
    z_pregrasp: float | None = None
    if z_safe is not None:
        lines.append(f"- z_high (first approach above pick target): z >= {z_safe:.3f}")
    if manip:
        z_obj_top = max(z for _n, _x, _y, z in manip)
        z_pregrasp = z_obj_top + min(hover_m * 0.5, 0.08)
        if z_safe is not None and z_pregrasp >= z_safe - 0.02:
            z_pregrasp = max(z_obj_top + 0.04, z_safe - 0.10)
        lines.append(
            f"- z_pregrasp (second approach, closer before grasp descend): z >= {z_pregrasp:.3f} "
            f"at pick xy from PERCEPTION"
        )
        lines.append(
            f"- Pick target xy hint: x≈{pick[1]:.3f}, y≈{pick[2]:.3f} (use PERCEPTION for exact values)"
        )
    if tcps and manip and z_safe is not None:
        arm = "r" if pick[2] < 0 else "l"
        tcp = tcps.get(arm) or next(iter(tcps.values()))
        lines.append("")
        lines.append("MANDATORY MOTION ORDER (planner must output these steps before pick approach):")
        lines.append(
            f"- Step 1 — Lift only (first subtask): xyz "
            f"[{tcp[0]:.4f}, {tcp[1]:.4f}, {z_safe:.3f}], rpy_deg [0, 0, 0] "
            f"(keep current TCP x,y; raise z to z_high only)."
        )
        lines.append(
            f"- Step 2 — Transit only (second subtask): xyz "
            f"[{pick[1]:.4f}, {pick[2]:.4f}, {z_safe:.3f}], rpy_deg [0, 0, 0] "
            f"(move xy to pick target; z stays z_high)."
        )
        if z_pregrasp is not None:
            lines.append(
                f"- Step 3a — Above pick (high): xyz [{pick[1]:.4f}, {pick[2]:.4f}, {z_safe:.3f}], rpy_deg [0, 0, 0]"
            )
            lines.append(
                f"- Step 3b — Above pick (pre-grasp): xyz [{pick[1]:.4f}, {pick[2]:.4f}, {z_pregrasp:.3f}], rpy_deg [0, 0, 0]"
            )
        lines.append(
            "- Step 4 — Grasp: descend z to object from PERCEPTION at pick xy, then close gripper."
        )
        lines.append(
            "- Never combine step 1 and step 2 in one goto_pose (diagonal path hits the table)."
        )
    if tcps and manip:
        lines.append(
            "- Top-down rpy_deg for every goto_pose in pick-and-place: [0, 0, 0] "
            "(fingers-down on hardware). ROBOT_STATE rpy is FK readback only — do not copy it into plans."
        )
        lines.append(
            "- Parallel-to-ground (side grasp): pitch≈-85° to -90°; roll/yaw depend on arm pose — "
            "do not use for vertical pick-and-place."
        )
    if tcps:
        tz_min = min(t[2] for t in tcps.values())
        lines.append(
            f"- TCP z≈{tz_min:.2f}: must reach z_high before any xy move toward objects."
        )
    lines.append(
        "- One goto_pose blends xyz and rpy: do not switch to pitch≈-90° during xy transit (that reorients gripper horizontal)."
    )
    return "\n".join(lines)
