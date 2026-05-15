"""Parse LLM JSON plans and dispatch guarded Reachy SDK motions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from reachy2_sdk.utils.utils import get_pose_matrix

from reachy_system2.config import SafeWorkspace

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    ok: bool
    message: str
    out_of_workspace: bool = False
    rejected_xyz: tuple[float, float, float] | None = None


def validate_xyz_in_workspace(ws: SafeWorkspace, xyz: tuple[float, float, float]) -> ExecutionResult:
    x, y, z = xyz
    if ws.contains(x, y, z):
        return ExecutionResult(True, "in_bounds")
    return ExecutionResult(
        False,
        f"out_of_workspace: ({x:.4f},{y:.4f},{z:.4f}) not inside safe box",
        out_of_workspace=True,
        rejected_xyz=(x, y, z),
    )


class ActionExecutor:
    """Structured action dispatcher with workspace guardrails."""

    def __init__(self, reachy: Any, workspace: SafeWorkspace | None = None) -> None:
        self.reachy = reachy
        self.workspace = workspace or SafeWorkspace.from_env()

    def _check_pose_xyz(self, xyz: list | tuple) -> ExecutionResult:
        if len(xyz) != 3:
            return ExecutionResult(False, "xyz must have length 3")
        t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        return validate_xyz_in_workspace(self.workspace, t)

    def run_action(self, action: dict[str, Any], *, wait: bool = True) -> ExecutionResult:
        if not isinstance(action, dict) or "op" not in action:
            return ExecutionResult(False, "action must be a dict with 'op'")
        op = action["op"]

        if op == "r_arm_goto_pose":
            chk = self._check_pose_xyz(action["xyz"])
            if not chk.ok:
                return chk
            rpy = action.get("rpy_deg", [0, 0, 0])
            dur = float(action.get("duration", 3.0))
            pose = get_pose_matrix(list(action["xyz"]), list(rpy))
            self.reachy.r_arm.goto(
                pose,
                duration=dur,
                wait=wait,
                interpolation_space="cartesian_space",
                interpolation_mode="minimum_jerk",
            )
            return ExecutionResult(True, "r_arm_goto_pose ok")

        if op == "l_arm_goto_pose":
            chk = self._check_pose_xyz(action["xyz"])
            if not chk.ok:
                return chk
            rpy = action.get("rpy_deg", [0, 0, 0])
            dur = float(action.get("duration", 3.0))
            pose = get_pose_matrix(list(action["xyz"]), list(rpy))
            self.reachy.l_arm.goto(
                pose,
                duration=dur,
                wait=wait,
                interpolation_space="cartesian_space",
                interpolation_mode="minimum_jerk",
            )
            return ExecutionResult(True, "l_arm_goto_pose ok")

        if op == "r_arm_translate":
            self.reachy.r_arm.translate_by(
                float(action.get("x", 0.0)),
                float(action.get("y", 0.0)),
                float(action.get("z", 0.0)),
                frame=str(action.get("frame", "robot")),
                wait=wait,
            )
            return ExecutionResult(True, "r_arm_translate ok")

        if op == "l_arm_translate":
            self.reachy.l_arm.translate_by(
                float(action.get("x", 0.0)),
                float(action.get("y", 0.0)),
                float(action.get("z", 0.0)),
                frame=str(action.get("frame", "robot")),
                wait=wait,
            )
            return ExecutionResult(True, "l_arm_translate ok")

        if op == "r_arm_rotate":
            self.reachy.r_arm.rotate_by(
                float(action.get("roll", 0.0)),
                float(action.get("pitch", 0.0)),
                float(action.get("yaw", 0.0)),
                frame=str(action.get("frame", "gripper")),
                wait=wait,
            )
            return ExecutionResult(True, "r_arm_rotate ok")

        if op == "l_arm_rotate":
            self.reachy.l_arm.rotate_by(
                float(action.get("roll", 0.0)),
                float(action.get("pitch", 0.0)),
                float(action.get("yaw", 0.0)),
                frame=str(action.get("frame", "gripper")),
                wait=wait,
            )
            return ExecutionResult(True, "l_arm_rotate ok")

        if op == "r_gripper_goto":
            pos = int(action.get("position", 0))
            dur = float(action.get("duration", 2.0))
            self.reachy.r_arm.gripper.goto(pos, duration=dur, interpolation_mode="minimum_jerk", wait=wait)
            return ExecutionResult(True, "r_gripper_goto ok")

        if op == "l_gripper_goto":
            pos = int(action.get("position", 0))
            dur = float(action.get("duration", 2.0))
            self.reachy.l_arm.gripper.goto(pos, duration=dur, interpolation_mode="minimum_jerk", wait=wait)
            return ExecutionResult(True, "l_gripper_goto ok")

        return ExecutionResult(False, f"unknown op: {op}")

    def run_subtask(
        self,
        subtask: dict[str, Any],
        *,
        wait: bool = True,
        dry_run: bool = False,
    ) -> ExecutionResult:
        acts = subtask.get("actions", [])
        if not isinstance(acts, list):
            return ExecutionResult(False, "subtask.actions must be a list")
        if len(acts) > 5:
            return ExecutionResult(False, "subtask has more than 5 actions")
        for i, a in enumerate(acts):
            logger.info("action %s/%s: %s", i + 1, len(acts), a)
            if dry_run:
                continue
            res = self.run_action(a, wait=wait)
            if not res.ok:
                return res
        return ExecutionResult(True, "subtask complete")

    def validate_subtask_bounds(self, subtask: dict[str, Any]) -> ExecutionResult:
        """Pre-flight check all Cartesian targets in a subtask."""
        for a in subtask.get("actions", []):
            if not isinstance(a, dict):
                continue
            if a.get("op") in ("r_arm_goto_pose", "l_arm_goto_pose"):
                chk = self._check_pose_xyz(a["xyz"])
                if not chk.ok:
                    return chk
        return ExecutionResult(True, "all targets in workspace")
