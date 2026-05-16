"""Summarize live robot state for LLM prompts (end-effector poses, grippers)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _tcp_xyz_from_fk(fk: Any) -> tuple[float, float, float]:
    m = np.asarray(fk, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix, got shape {m.shape}")
    t = m[:3, 3]
    return float(t[0]), float(t[1]), float(t[2])


def format_end_effector_context_for_llm(reachy: Any) -> str:
    """Build a text block: TCP positions in robot base frame + gripper openings.

    Uses ``forward_kinematics()`` as in ``3_arm_and_gripper.ipynb``. Missing arms are omitted.
    """
    lines: list[str] = [
        "ROBOT_STATE (robot base frame, meters; same axes as PERCEPTION — +X forward, +Y robot LEFT, +Z up):",
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
            any_arm = True
            lines.append(f"- {label} gripper TCP position: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            gripper = getattr(arm, "gripper", None)
            if gripper is not None:
                opening = getattr(gripper, "present_position", None)
                if opening is None:
                    opening = getattr(gripper, "opening", None)
                if opening is not None:
                    lines.append(f"  gripper present opening (device units, larger ≈ more open): {opening}")
        except Exception as e:
            logger.warning("%s forward_kinematics failed: %s", attr, e)
            lines.append(f"- {label}: pose unavailable ({e})")

    if not any_arm:
        lines.append("(no arm kinematics available)")
    return "\n".join(lines)
