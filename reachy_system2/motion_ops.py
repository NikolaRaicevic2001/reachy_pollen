"""Motion operation descriptions for LLM prompts."""

MOTION_OPS_FOR_LLM = """AVAILABLE_OPERATIONS (exact op names; prefix r_ or l_ for right/left arm):

Reachy rpy_deg order: [roll, pitch, yaw] in degrees (robot base frame), intrinsic rotations about X, then Y, then Z (SDK get_pose_matrix: Rz @ Ry @ Rx).
- roll / pitch / yaw — top-down desk grasp: use rpy_deg [0, 0, 0] for lift, transit, descend, and place (hardware-verified). pitch ≈ -90° is gripper parallel to the ground (side grasp), NOT top-down.
Do not use pitch ≈ -90° or ±180° for a vertical pick from above the table.

r_arm_goto_pose / l_arm_goto_pose
  Absolute TCP pose. Position and orientation move together on one straight path in (xyz, rpy).
  Fields: xyz [m], rpy_deg [roll, pitch, yaw], duration [s].
  Do not use one goto_pose for lift+xy toward an object — follow SCENE_HINTS steps 1 then 2 separately.

r_arm_translate / l_arm_translate
  Relative translation only (orientation unchanged). Fields: x, y, z [m], frame "robot" or "gripper".

r_arm_rotate / l_arm_rotate
  Relative rotation only (position unchanged). Fields: roll, pitch, yaw deltas [deg], frame "robot" or "gripper".

r_gripper_goto / l_gripper_goto
  Gripper open/close only. position 0=closed, 100=open; duration [s]."""


MOBILE_BASE_OPS_FOR_LLM = """

MOBILE BASE (Reachy is mounted on a mobile base; movements are in the odometry frame):

mobile_base_translate_by
  Relative translation in meters. Fields: x, y [m], wait [bool], timeout [s].
  Example: {"op":"mobile_base_translate_by","x":0.2,"y":0.0,"wait":true,"timeout":5}

mobile_base_rotate_by
  Relative rotation in degrees. Fields: theta [deg], wait [bool], timeout [s].
  Example: {"op":"mobile_base_rotate_by","theta":-90.0,"wait":true}

Notes:
- Use mobile base moves to change viewpoint when required objects are not visible yet (e.g. “on your right”, “behind”, “on a chair”).
- After a base move, expect PERCEPTION to change; re-check detections before planning arm motions.
"""
