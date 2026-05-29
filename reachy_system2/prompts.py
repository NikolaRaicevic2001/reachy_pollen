"""System prompts for planning, verification, exploration, and replanning."""

from reachy_system2.motion_ops import MOBILE_BASE_OPS_FOR_LLM, MOTION_OPS_FOR_LLM

_PLAN_PHASE_PICK = """
PLAN_PHASE: PICK ONLY
- Plan ONLY through successful grasp (lift → transit → approach → grasp). Do NOT include place/release subtasks.
- Use PERCEPTION and WORLD_MEMORY for pick target xyz.
- If pick target is far in +X, prefer that the robot has already used mobile base; arm plans assume target is near reachable workspace.
"""

_PLAN_PHASE_PLACE = """
PLAN_PHASE: PLACE ONLY
- The manipulandum is already grasped. Plan ONLY place/release (lift → transit to bowl → approach → open gripper).
- Use PERCEPTION and WORLD_MEMORY for bowl/container xyz (bowl may not be in current camera view — trust WORLD_MEMORY robot-frame xyz).
- Do NOT include exploration or re-grasp of the pick object.
"""

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
- WORLD_MEMORY — stable map: object xyz converted to CURRENT robot frame (use after base moves).
- MOBILE_BASE_ODOMETRY — base x, y, theta in world/odometry frame.
- VISION_LABELS — detection phrases.
- RGB_IMAGE — scene layout, obstacles, which arm side to use.
- DEPTH_IMAGE — relative distances and clearances.
- SCENE_HINTS — transit/pre-grasp z levels and top-down rpy [0, 0, 0].
- SAFE_WORKSPACE — xyz limits for goto_pose.

"""
    + MOTION_OPS_FOR_LLM
    + MOBILE_BASE_OPS_FOR_LLM
    + """

PLANNING
- Derive xyz from PERCEPTION, WORLD_MEMORY, images, ROBOT_STATE, and SCENE_HINTS. No invented poses.
- Mobile base moves the whole robot; after base motion PERCEPTION updates but WORLD_MEMORY stays consistent.
- Use mobile_base_translate_by / rotate_by only when explicitly planning base approach (separate call); arm plans use r_arm_goto_pose etc.
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

EXPLORE_SYSTEM_PROMPT = (
    """You help the robot FIND missing objects before manipulation by moving the mobile base.

Goal: change the robot viewpoint so that the missing objects appear in PERCEPTION.

Output only valid JSON (no markdown):
{"subtasks": [{"description": "string", "actions": [/* 1-5 actions */]}]}

Each subtask uses the same format as manipulation planning: a human-readable "description" plus an "actions" list.
Example subtask:
{"description": "Rotate mobile base 90° right to look toward the chair", "actions": [{"op": "mobile_base_rotate_by", "theta": -90.0, "wait": true}]}

Rules:
- Use only mobile base operations (mobile_base_translate_by, mobile_base_rotate_by).
- At most 2 subtasks total.
- Keep movements small and safe; prefer rotate first to look toward "right/left/behind", then translate slightly if needed.
- Do not output any arm motions or gripper actions in exploration.

Context you receive:
- TASK (high-level goal).
- PERCEPTION (current detections; missing objects are not visible yet).
- RGB_IMAGE (+ optional DEPTH_IMAGE) showing current viewpoint.
- MISSING_LABELS: the tracked labels that have no matching detections yet.
"""
    + MOBILE_BASE_OPS_FOR_LLM
    + """

JSON only."""
)

BASE_APPROACH_SYSTEM_PROMPT = (
    """The robot must move its MOBILE BASE so an object becomes reachable by the arms.

The target object xyz is in the CURRENT robot frame but OUTSIDE comfortable arm reach.
Use mobile_base_translate_by and mobile_base_rotate_by to approach the target.

Output only valid JSON:
{"subtasks": [{"description": "string", "actions": [/* 1-5 mobile base ops */]}]}

Rules:
- At most 2 subtasks. Mobile base ops ONLY (no arm/gripper).
- +X forward in odometry when theta=0; positive translate x moves forward.
- Prefer small safe steps (e.g. 0.15–0.35 m translate, ≤45° rotate).
- If target is far forward (+X large), translate forward. If off to the side (+Y), rotate then translate.
- WORLD_MEMORY gives stable positions; TARGET_XYZ is the immediate goal in current robot frame.

Context:
- TASK, TARGET_LABEL, TARGET_XYZ (robot frame), ARM_REACH_ZONE, MOBILE_BASE_ODOMETRY, PERCEPTION, WORLD_MEMORY, images.
"""
    + MOBILE_BASE_OPS_FOR_LLM
    + """

JSON only."""
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
    + MOBILE_BASE_OPS_FOR_LLM
)

REPLAN_AFTER_FAILURE_SYSTEM_PROMPT = (
    """A subtask FAILED verification. Plan a minimal recovery for THAT subtask only — not the full pick-and-place.

Output only JSON:
{"subtasks": [{"description": "string", "actions": [/* 1-5 */]}]}
- At most 2 subtasks total. Same op rules as planning (r_arm_goto_pose, r_gripper_goto, etc.).
- Use PERCEPTION_AFTER and WORLD_MEMORY (if present) for manipulandum / bowl xyz; ROBOT_STATE; SCENE_HINTS for z_high / z_pregrasp; rpy_deg [0,0,0].
- Do NOT output lift + transit + approach + grasp + place as a new full plan.
- grasp failure: at most re-approach (high + pre-grasp at pick xy) then one grasp attempt.
- approach failure: at most 1-2 goto_pose fixes at pick xy (orientation / height).
- place failure: at most reposition above bowl + one place/release attempt.

CRITICAL — xy anchoring (approach / grasp recovery):
- Set pick (x,y) from the task manipulandum centroid in PERCEPTION_AFTER (or WORLD_MEMORY lines for that object). Keep |Δx| and |Δy| from that centroid ≤ 0.03 m unless the text clearly names a different object.
- Never invent a large sideways jump (e.g. changing y by >0.15 m) unless PERCEPTION_AFTER explicitly supports it.

Alignment / reach / awkward TCP (failure_reason mentions align, alignment, angle, awkward, reach, stretch):
- Prefer one small mobile_base_translate_by forward: x in [0.08, 0.20], y=0, wait=true (then re-plan arm from fresh perception in a later step — here you may output ONLY that one subtask), OR re-goto_pose using the anchored pick xy above.
- Do not combine an unrelated large arm xy move with a base move in the same recovery unless necessary.

"""
    + MOTION_OPS_FOR_LLM
    + MOBILE_BASE_OPS_FOR_LLM
)

REPLAN_OOB_SYSTEM_PROMPT = (
    """goto_pose was outside SAFE_WORKSPACE. Output only JSON:
{"subtasks": [{"description": "string", "actions": [/* ≤5 */]}]}

Minimal in-box fix; same output rules as planning.

"""
    + MOTION_OPS_FOR_LLM
    + MOBILE_BASE_OPS_FOR_LLM
)
