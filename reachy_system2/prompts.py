"""System prompts for planning, verification, and out-of-workspace replanning."""

from reachy_system2.motion_ops import MOTION_OPS_FOR_LLM

LABELS_SYSTEM_PROMPT = """From the TASK, list short noun phrases for vision tracking (OWL-ViT).

Output only JSON: {"labels": ["...", "..."]}

Rules
- 1-8 simple phrases the detector can match.
- Include task objects and fixed scene elements (e.g. table).
- Exclude the robot."""

PLAN_SYSTEM_PROMPT = (
    """You plan motion for a Reachy 2 robot. Output only valid JSON (no markdown).

Coordinate frame (robot base, meters): +X forward, +Y robot left, +Z up.

OUTPUT (strict)
{"subtasks": [{"description": "string", "actions": [/* 1-5 actions */]}]}
- Use full op names (e.g. r_arm_goto_pose). Flat actions only; required field "op".
- Group related steps in one subtask (several actions), not one tiny action per subtask.
- Typical pick-and-place: about 5-7 subtasks. Never skip the mandatory opening steps in SCENE_HINTS.

USER MESSAGE INPUTS (use all of these — do not invent poses without them)
- TASK — goal for this run.
- ROBOT_STATE — both arms: TCP xyz; rpy_deg is FK readback only (not for goto_pose).
- PERCEPTION — object poses in robot frame (primary source for pick/place xyz).
- VISION_LABELS — detection phrases.
- RGB_IMAGE — scene layout, obstacles, which arm side to use.
- DEPTH_IMAGE — relative distances and clearances.
- SCENE_HINTS — transit/pre-grasp z levels and top-down rpy [0, 0, 0].
- SAFE_WORKSPACE — xyz limits for goto_pose.

"""
    + MOTION_OPS_FOR_LLM
    + """

PLANNING
- Derive xyz from PERCEPTION, images, ROBOT_STATE, and SCENE_HINTS. No invented poses.
- Top-down: rpy_deg [0, 0, 0] on every goto_pose. Do not copy ROBOT_STATE rpy. Do not use pitch ≈ -90°.
- goto_pose moves in a straight line in (xyz, rpy). Never combine a large z change with a large xy change in one action (scrapes the table).

MANDATORY ORDER (SCENE_HINTS lists exact numbers — follow before any move toward the pick target):
1. First subtask — Lift only: at step-1 xyz (current arm TCP xy, z = z_high). Same x and y as ROBOT_STATE for the working arm; only z increases.
2. Second subtask — Transit only: at step-2 xyz (pick target x,y from PERCEPTION, z = z_high). xy changes, z stays at z_high.
3. Third subtask — Approach pick: at pick xy, two goto_pose actions — (a) z = z_high, (b) z = z_pregrasp. Both [0, 0, 0].
4. Grasp: descend to grasp z at pick xy, then close gripper.
5. Place path: lift to z_high at pick xy → transit at z_high to bowl xy → approach (z_high then z_pregrasp) → descend and open gripper.

Do not start the plan with a move to the object. Do not go straight to the can/bowl from the home pose.

Complete the TASK. JSON only."""
)

VERIFY_SYSTEM_PROMPT = (
    """Check if the attempted subtask succeeded using BEFORE/AFTER images, PERCEPTION_AFTER, and SUBTASK_MODE.

Judge only the stated subtask — not the whole task. Use SUBTASK_MODE criteria:
- approach: TCP above object xy, gripper fingers-down (not horizontal). Do NOT require the object to be lifted.
- grasp: manipulandum grasped or lifted; gripper closed on the object.
- place: object over target container; gripper opened after release.

Output only JSON:
{"status": "OK"|"FAILED", "failure_reason": "short string or null", "correction": null | {"description": "string", "actions": [/* ≤5 */]}}

Corrections must use full op names (e.g. r_arm_goto_pose, r_arm_translate, r_gripper_goto) — never "operation" or bare x/y/z.
If FAILED and needs re-approach or full re-grasp sequence, set correction to null (recovery replan handles that).

"""
    + MOTION_OPS_FOR_LLM
)

REPLAN_AFTER_FAILURE_SYSTEM_PROMPT = (
    """A subtask FAILED verification. Plan a minimal recovery for THAT subtask only — not the full pick-and-place.

Output only JSON:
{"subtasks": [{"description": "string", "actions": [/* 1-5 */]}]}
- At most 2 subtasks total. Same op rules as planning (r_arm_goto_pose, r_gripper_goto, etc.).
- Use PERCEPTION_AFTER for object xyz; ROBOT_STATE; SCENE_HINTS for z_high / z_pregrasp; rpy_deg [0,0,0].
- Do NOT output lift + transit + approach + grasp + place as a new full plan.
- grasp failure: at most re-approach (high + pre-grasp at pick xy) then one grasp attempt.
- approach failure: at most 1-2 goto_pose fixes at pick xy (orientation / height).
- place failure: at most reposition above bowl + one place/release attempt.

"""
    + MOTION_OPS_FOR_LLM
)

REPLAN_OOB_SYSTEM_PROMPT = (
    """goto_pose was outside SAFE_WORKSPACE. Output only JSON:
{"subtasks": [{"description": "string", "actions": [/* ≤5 */]}]}

Minimal in-box fix; same output rules as planning.

"""
    + MOTION_OPS_FOR_LLM
)
