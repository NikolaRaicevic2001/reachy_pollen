"""Rules for when vision verification and recovery replan apply to a subtask."""

from __future__ import annotations


def subtask_verification_mode(description: str) -> str:
    """Classify a planned subtask for verification criteria.

    Returns one of: kinematic, approach, grasp, place.
    """
    d = description.lower()
    if any(w in d for w in ("mobile base", "mobile_base", "rotate", "translation", "translate", "move base", "turn base")):
        return "kinematic"
    if any(w in d for w in ("grasp", "grip the", "close gripper", "pick up the", "pick the")):
        return "grasp"
    if any(w in d for w in ("place", "release", "open gripper", "drop", "into the", "into the bowl")):
        return "place"
    if any(w in d for w in ("approach", "pre-grasp", "pregrasp", "pre grasp")):
        return "approach"
    return "kinematic"


def skip_vision_verification(mode: str) -> bool:
    """Lift / transit / z_high moves — execution success is enough; vision cannot judge these."""
    return mode == "kinematic"


def allows_failure_recovery(mode: str) -> bool:
    """Only re-grasp / re-approach / place recovery — never replace the whole pick-and-place plan."""
    return mode in ("grasp", "place", "approach")
