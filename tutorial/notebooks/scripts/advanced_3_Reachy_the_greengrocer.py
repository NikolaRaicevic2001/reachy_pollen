"""Reachy the greengrocer: pick-and-place with Pollen-Vision.

Converted from advanced_3_Reachy_the_greengrocer.ipynb.
Run: python advanced_3_reachy_greengrocer.py

Live detection: continuous video feed. ENTER = accept positions, Q/ESC = abort.
Phase gates (MANUAL_GATE=True): ENTER to continue each step; Q to abort.
Pre-grasp / pre-drop: image window with target highlighted; ENTER to proceed.
Manipulation loop: Ctrl+C or Q at a gate to stop.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from reachy2_sdk import ReachySDK
from reachy2_sdk.utils.utils import get_pose_matrix, invert_affine_transformation_matrix
from pollen_vision.camera_wrappers.pollen_sdk_camera.pollen_sdk_camera_wrapper import (
    PollenSDKCameraWrapper,
)
from pollen_vision.perception import Perception
from pollen_vision.vision_models.object_detection import OwlVitWrapper
from pollen_vision.utils import Annotator
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBOT_HOST = "192.168.137.100"
MUJOCO_MODE = False
MANUAL_GATE = False
GATE_WINDOW = "greengrocer - confirm"

# Picking targets and where they go (None on either side disables that arm).
PICK_LEFT = "cucumber"
PICK_RIGHT = "cylindrical can"
GOAL_LEFT = "metal bowl"
GOAL_RIGHT = "rectangular basket"

PERCEPTION_FREQ = 40
DETECTION_THRESHOLD = 0.15
MIN_DETECTION_THRESHOLD = 0.05
OWL_VIT_THRESHOLD = 0.05

DISTANCES = {
    "base_back_to_prep": 0.30,
    "base_back_on_finish": 0.30,
    "drop_above_goal": 0.22,
    "drop_descend": 0.10,
    "post_drop_lift": 0.15,
    "post_grasp_lift": 0.15,
    "pregrasp_offset_x": 0.08,
    "torso_safety_x": 0.15,
    "torso_safety_y": 0.15,
    "object_in_target_radius": 0.15,
    "fruit_radius": 0.02,
    "gripper_finger_offset": 0.04,
    "gripper_y_inset": 0.01,
}

WAITING_POSE_LEFT_JOINTS = [30, -10, 15, -115, 0, 0, 15]
WAITING_POSE_RIGHT_JOINTS = [30, 10, -15, -115, 0, 0, -15]
HEAD_LOOK_OFFSET_X = 0.3

BENT_POSITION_LEFT = np.array([0.38622, 0.22321, -0.27036])
BENT_POSITION_RIGHT = np.array([0.38622, -0.22321, -0.27036])

LIVE_DETECTION_WINDOW = "greengrocer - live detection"


# ---------------------------------------------------------------------------
# Gripper wait (MuJoCo vs real robot)
# ---------------------------------------------------------------------------


if MUJOCO_MODE:

    def wait_for_gripper(arm) -> None:
        while arm.gripper.is_moving():
            time.sleep(0.1)
        time.sleep(1)

else:

    def wait_for_gripper(arm) -> None:
        while arm.gripper.is_moving():
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Robot poses and motion helpers
# ---------------------------------------------------------------------------


def get_to_waiting_pose(reachy: ReachySDK, duration: float = 2.0) -> None:
    """Move arms to parallel-to-ground waiting pose and look at the work area."""
    print(f"[pose] Moving to waiting pose (duration={duration}s)...")
    reachy.r_arm.goto(WAITING_POSE_RIGHT_JOINTS, duration)
    reachy.l_arm.goto(WAITING_POSE_LEFT_JOINTS, duration)
    reachy.r_arm.gripper.open()
    reachy.l_arm.gripper.open()
    waiting_position = reachy.r_arm.forward_kinematics(WAITING_POSE_RIGHT_JOINTS)[:3, 3]
    reachy.head.look_at(
        waiting_position[0] + HEAD_LOOK_OFFSET_X,
        0,
        waiting_position[2],
        duration,
        wait=True,
    )
    print("[pose] Reachy in waiting pose.")


def distance_between_objects(
    obj1: npt.NDArray[np.float64], obj2: npt.NDArray[np.float64]
) -> float:
    return float(np.linalg.norm(obj1[:3, 3] - obj2[:3, 3]))


def object_in_target(obj: Dict, target: Dict, threshold: float) -> bool:
    return distance_between_objects(obj["pose"], target["pose"]) < threshold


def is_a_new_object(actual_obj: Dict, former_obj: Dict, threshold: float) -> bool:
    if not former_obj:
        return True
    return distance_between_objects(actual_obj["pose"], former_obj["pose"]) > threshold


def is_too_close(obj: Dict) -> bool:
    x_lim = DISTANCES["torso_safety_x"]
    y_lim = DISTANCES["torso_safety_y"]
    if obj["pose"][0, 3] < x_lim and abs(obj["pose"][1, 3]) < y_lim:
        print(f"[safety] Object {obj['name']} at {obj['pose'][:3, 3]} is too close to the robot.")
        return True
    return False


def get_selected_objects(
    perception: Perception,
    obj_to_left: str,
    obj_to_right: str,
    left_obj_target: str,
    right_obj_target: str,
    detection_threshold: float,
) -> Tuple[List[dict], List[dict], dict, dict]:
    """Detect pick targets and goal containers; lower threshold until goals are found."""
    print(f"[detect] Scanning for objects (threshold={detection_threshold})...")
    detected_objects = perception.get_objects_infos(detection_threshold)

    objects_to_left = [obj for obj in detected_objects if obj["name"] == obj_to_left]
    objects_to_right = [obj for obj in detected_objects if obj["name"] == obj_to_right]

    min_threshold = MIN_DETECTION_THRESHOLD
    while True:
        left_target = [obj for obj in detected_objects if obj["name"] == left_obj_target]
        right_target = [obj for obj in detected_objects if obj["name"] == right_obj_target]

        if left_target and right_target:
            break

        if detection_threshold > min_threshold:
            detection_threshold -= 0.01
            print(f"[detect] Targets not found, lowering threshold to {detection_threshold:.2f}")
        else:
            print("[detect] Targets not found at minimal threshold, retrying...")
        detected_objects = perception.get_objects_infos(detection_threshold)
        time.sleep(1.5)

    print(
        f"[detect] {len(detected_objects)} objects total: "
        f"{len(objects_to_left)} {obj_to_left}, {len(objects_to_right)} {obj_to_right}, "
        f"1 {left_obj_target}, 1 {right_obj_target}."
    )
    return objects_to_left, objects_to_right, left_target[0], right_target[0]


def get_closest_object(
    reachy: ReachySDK,
    former_object: Dict,
    obj_to_left: List[dict],
    obj_to_right: List[dict],
    left_target: Dict,
    right_target: Dict,
    dist_threshold: float,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Return the closest graspable pick object and which arm ('left' or 'right') to use."""
    closest_object: Optional[Dict] = None
    side: Optional[str] = None
    min_dist = np.inf
    fruit_radius = DISTANCES["fruit_radius"]

    def find_closest(objects: List[dict], effector_pos: npt.NDArray, side_name: str) -> None:
        nonlocal closest_object, min_dist, side
        for obj in objects:
            dist_with_effector = distance_between_objects(obj["pose"], effector_pos)
            if (
                dist_with_effector < min_dist
                and not object_in_target(obj, left_target, dist_threshold)
                and not object_in_target(obj, right_target, dist_threshold)
                and not is_too_close(obj)
                and is_a_new_object(obj, former_object, fruit_radius)
            ):
                closest_object = obj
                min_dist = dist_with_effector
                side = side_name

    position_l_effector = reachy.l_arm.forward_kinematics()
    position_r_effector = reachy.r_arm.forward_kinematics()

    if left_target:
        find_closest(obj_to_left, position_l_effector, "left")
    if right_target:
        find_closest(obj_to_right, position_r_effector, "right")

    if closest_object:
        print(
            f"[pick] Closest object: {closest_object['name']} "
            f"at {closest_object['pose'][:3, 3]} (arm={side})"
        )
    else:
        print("[pick] No graspable object in the workspace.")

    return closest_object, side


def get_goal_pose(object_pose_dict: Dict, target_side: str) -> npt.NDArray[np.float64]:
    """Build a 4x4 grasp/drop pose from an object detection dict."""
    bent_position = BENT_POSITION_LEFT if target_side == "left" else BENT_POSITION_RIGHT
    object_pose = np.copy(object_pose_dict["pose"])

    finger = DISTANCES["gripper_finger_offset"]
    y_inset = DISTANCES["gripper_y_inset"]
    object_pose[0, 3] -= finger
    if target_side == "left":
        object_pose[1, 3] += y_inset
    else:
        object_pose[1, 3] -= y_inset
    object_pose[2, 3] -= finger

    dy = object_pose[1, 3] - bent_position[1]
    dx = object_pose[0, 3]
    angle_x_rad = np.arctan2(dx, dy)
    angle_x = 90 - np.degrees(angle_x_rad)
    target_pose = get_pose_matrix(object_pose[:3, 3], [angle_x, -90, 0])
    print(f"[ik] Goal position {target_pose[:3, 3]}, rotation {[angle_x, -90, 0]}")
    return target_pose


def get_pregrasping_pose(
    goal_pose: npt.NDArray[np.float64], target_side: str
) -> npt.NDArray[np.float64]:
    """Pose slightly before the grasp pose along the approach axis."""
    bent_position = BENT_POSITION_LEFT if target_side == "left" else BENT_POSITION_RIGHT
    pregrasp_pose = goal_pose.copy()
    rotation = R.from_matrix(goal_pose[:3, :3]).as_euler("xyz", degrees=False)[0]
    pregrasp_pose[0, 3] -= DISTANCES["pregrasp_offset_x"]
    pregrasp_pose[1, 3] = pregrasp_pose[0, 3] * np.tan(rotation) + bent_position[1]
    return pregrasp_pose


def move_head(reachy: ReachySDK, target_obj: Dict) -> None:
    target_pose = target_obj["pose"]
    print(f"[head] Looking at {target_obj['name']} at {target_pose[:3, 3]}")
    reachy.head.look_at(target_pose[0, 3], target_pose[1, 3], target_pose[2, 3], 2)


def move_to_grasp(reachy: ReachySDK, obj_to_catch: Dict, target_side: str) -> None:
    print("[grasp] Move sequence started.")
    move_head(reachy, obj_to_catch)

    grasp_pose = get_goal_pose(obj_to_catch, target_side)
    pregrasp_pose = get_pregrasping_pose(grasp_pose, target_side)

    arm = reachy.l_arm if target_side == "left" else reachy.r_arm
    print("[grasp] Computing inverse kinematics for pregrasp and grasp...")
    joints_to_pregrasp = arm.inverse_kinematics(pregrasp_pose)
    joints_to_grasp = arm.inverse_kinematics(grasp_pose)

    print("[grasp] Moving to pregrasp pose...")
    arm.goto(joints_to_pregrasp)
    print("[grasp] Moving to grasp pose...")
    arm.goto(joints_to_grasp, wait=True)

    print("[grasp] Closing gripper...")
    arm.gripper.close()
    wait_for_gripper(arm)
    lift = DISTANCES["post_grasp_lift"]
    print(f"[grasp] Lifting object by {lift}m...")
    arm.translate_by(x=0, y=0, z=lift, duration=1)
    time.sleep(1)
    print("[grasp] Move sequence done.")


def move_to_drop(reachy: ReachySDK, target_side: str, goal_obj: Dict) -> None:
    print("[drop] Move sequence started.")
    arm = reachy.l_arm if target_side == "left" else reachy.r_arm
    target_obj = copy.deepcopy(goal_obj)
    target_obj["pose"][2, 3] += DISTANCES["drop_above_goal"]
    print(
        f"[drop] Target {target_obj['name']} raised to "
        f"{target_obj['pose'][:3, 3]} (above goal)"
    )

    move_head(reachy, target_obj)
    drop_pose = get_goal_pose(target_obj, target_side)
    print("[drop] Computing inverse kinematics for drop pose...")
    joints_to_drop = arm.inverse_kinematics(drop_pose)

    print("[drop] Moving above goal...")
    arm.goto(joints_to_drop)
    descend = DISTANCES["drop_descend"]
    print(f"[drop] Descending {descend}m...")
    arm.translate_by(0, 0, -descend, duration=1, wait=True)

    print("[drop] Opening gripper...")
    arm.gripper.open()
    wait_for_gripper(arm)
    lift = DISTANCES["post_drop_lift"]
    print(f"[drop] Lifting arm by {lift}m...")
    arm.translate_by(0, 0, lift, duration=1, wait=True)
    print("[drop] Move sequence done.")


def make_no_from_head(reachy: ReachySDK) -> None:
    """Shake head left-right to signal unreachable / too far."""
    print("[feedback] Shaking head (no gesture)...")
    reachy.head.look_at(x=0.5, y=0.15, z=0.15, duration=1)
    reachy.head.look_at(x=0.5, y=-0.15, z=0.15, duration=1)
    reachy.head.look_at(x=0.5, y=0.15, z=0.15, duration=1)
    reachy.head.look_at(x=0.5, y=0, z=-0.25, duration=1, wait=True)
    print("[feedback] Head shake done.")


def target_object_unreachable(reachy: ReachySDK, target_side: str) -> None:
    """Recover when drop pose is unreachable: shake head, release object, return to wait."""
    print("[recover] Goal unreachable — releasing object and returning to waiting pose.")
    make_no_from_head(reachy)
    arm = reachy.l_arm if target_side == "left" else reachy.r_arm
    arm.translate_by(0, 0, -0.1, wait=True)
    arm.gripper.open()
    wait_for_gripper(arm)
    get_to_waiting_pose(reachy)


# ---------------------------------------------------------------------------
# Manual confirmation gates
# ---------------------------------------------------------------------------


def confirm(prompt: str) -> None:
    """Pause until ENTER. Type 'q' then ENTER to abort. No-op if MANUAL_GATE is False."""
    if not MANUAL_GATE:
        return
    print(f"\n[gate] {prompt}")
    reply = input("       Press ENTER to continue (or 'q' + ENTER to abort): ")
    if reply.strip().lower() in ("q", "quit", "abort"):
        raise KeyboardInterrupt("Aborted at confirmation prompt.")


def confirm_with_image(
    prompt: str, image_bgr: npt.NDArray[np.uint8], window_title: str = GATE_WINDOW
) -> None:
    """Show image and wait for ENTER (or Q/ESC to abort)."""
    if not MANUAL_GATE:
        return
    import cv2

    print(f"\n[gate] {prompt}")
    print("       ENTER = continue, Q/ESC = abort.")
    while True:
        cv2.imshow(window_title, image_bgr)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.destroyWindow(window_title)
            return
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_title)
            raise KeyboardInterrupt("Aborted at confirmation prompt.")


def build_target_preview(
    r_cam: PollenSDKCameraWrapper,
    owlvit: OwlVitWrapper,
    annotator: Annotator,
    labels: List[str],
    target: Dict,
    caption: str,
) -> npt.NDArray[np.uint8]:
    """Grab a fresh frame, annotate detections, highlight the chosen target. Returns BGR image."""
    import cv2

    data, _, _ = r_cam.get_data()
    img_rgb = data["left"]
    preds = owlvit.infer(
        im=img_rgb[:, :, ::-1],
        candidate_labels=labels,
        detection_threshold=OWL_VIT_THRESHOLD,
    )
    annotated_rgb = annotator.annotate(im=img_rgb, detection_predictions=preds)
    img = annotated_rgb[:, :, ::-1].copy()

    bbox = target.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 8, (0, 0, 255), -1)
    pose_xyz = target["pose"][:3, 3]
    label = (
        f"{caption}  {target['name']}  "
        f"xyz=({pose_xyz[0]:.2f},{pose_xyz[1]:.2f},{pose_xyz[2]:.2f})"
    )
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Live detection (continuous; ENTER / Q)
# ---------------------------------------------------------------------------


def live_detection_loop(
    r_cam: PollenSDKCameraWrapper,
    perception: Perception,
    owlvit: OwlVitWrapper,
    annotator: Annotator,
    labels: List[str],
) -> List[dict]:
    """Continuous annotated video feed. ENTER = accept, Q/ESC = abort."""
    import cv2

    print(
        "\n[live] Live detection started (continuous feed).\n"
        "       ENTER  = accept current detections and continue\n"
        "       Q/ESC  = abort\n"
    )

    latest_detections: List[dict] = []
    last_names: Optional[List[str]] = None

    while True:
        data, _, _ = r_cam.get_data()
        img = data["left"]

        latest_detections = perception.get_objects_infos(DETECTION_THRESHOLD)
        owlvit_predictions = owlvit.infer(
            im=img[:, :, ::-1],
            candidate_labels=labels,
            detection_threshold=OWL_VIT_THRESHOLD,
        )
        annotated = annotator.annotate(im=img, detection_predictions=owlvit_predictions)

        names = sorted(d.get("name", "?") for d in latest_detections)
        if names != last_names:
            print(f"[live] Perception tracked: {names if names else '(none)'}")
            last_names = names

        cv2.imshow(LIVE_DETECTION_WINDOW, annotated[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF

        if key in (13, 10):
            print("[live] ENTER pressed - accepting detections.")
            break
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Live detection aborted by user.")

    cv2.destroyAllWindows()
    return latest_detections


def build_perception_stack(reachy: ReachySDK) -> Tuple[
    PollenSDKCameraWrapper, Perception, OwlVitWrapper, Annotator, List[str]
]:
    """Create camera wrapper, Perception tracker, and OwlVit annotator."""
    print("[setup] Instantiating camera wrapper...")
    r_cam = PollenSDKCameraWrapper(reachy)

    print("[setup] Computing T_reachy_cam from depth extrinsics...")
    t_cam_reachy = reachy.cameras.depth.get_extrinsics()
    t_reachy_cam = invert_affine_transformation_matrix(t_cam_reachy)

    print(
        f"[setup] Creating Perception (freq={PERCEPTION_FREQ}, "
        f"threshold={DETECTION_THRESHOLD})..."
    )
    perception = Perception(
        r_cam,
        t_reachy_cam,
        freq=PERCEPTION_FREQ,
        detection_threshold=DETECTION_THRESHOLD,
    )

    labels = [x for x in [PICK_LEFT, PICK_RIGHT, GOAL_LEFT, GOAL_RIGHT] if x]
    print(f"[setup] Tracked labels: {labels}")
    perception.set_tracked_objects(labels)

    print("[setup] Loading OwlVit model (first run may take ~1 min)...")
    owlvit = OwlVitWrapper()
    annotator = Annotator()

    print("[setup] Starting Perception background tracking (visualize=False)...")
    perception.start(visualize=False)

    return r_cam, perception, owlvit, annotator, labels


def stop_perception(perception: Perception) -> None:
    if hasattr(perception, "stop"):
        print("[shutdown] Stopping Perception...")
        perception.stop()


def manipulation_loop(
    reachy: ReachySDK,
    perception: Perception,
    r_cam: PollenSDKCameraWrapper,
    owlvit: OwlVitWrapper,
    annotator: Annotator,
    labels: List[str],
) -> None:
    """Re-detect each iteration; grasp closest pick object and place in goal."""
    print("\n[loop] Starting grasp-and-place loop (Ctrl+C to stop).\n")
    former_object: Dict = {}

    while True:
        try:
            print("[loop] --- New iteration ---")
            left_obj, right_obj, left_goal, right_goal = get_selected_objects(
                perception,
                PICK_LEFT,
                PICK_RIGHT,
                GOAL_LEFT,
                GOAL_RIGHT,
                DETECTION_THRESHOLD,
            )

            closest_object, target_side = get_closest_object(
                reachy,
                former_object,
                left_obj,
                right_obj,
                left_goal,
                right_goal,
                DISTANCES["object_in_target_radius"],
            )

            if not closest_object:
                print("[loop] No reachable pick target; waiting 3s...")
                time.sleep(3)
                continue

            former_object = closest_object
            goal_obj = left_goal if target_side == "left" else right_goal
            print(f"[loop] Grasping {closest_object['name']} -> {goal_obj['name']} ({target_side} arm)")

            preview = build_target_preview(
                r_cam,
                owlvit,
                annotator,
                labels,
                target=closest_object,
                caption=f"GRASP ({target_side})",
            )
            confirm_with_image(
                f"About to grasp {closest_object['name']} with the {target_side} arm.",
                preview,
            )

            try:
                move_to_grasp(reachy, closest_object, target_side)
            except ValueError:
                print("[loop] Grasp pose unreachable — shaking head.")
                make_no_from_head(reachy)
                time.sleep(3)
                continue

            preview = build_target_preview(
                r_cam,
                owlvit,
                annotator,
                labels,
                target=goal_obj,
                caption=f"DROP ({target_side})",
            )
            confirm_with_image(
                f"About to drop into {goal_obj['name']} ({target_side} arm).",
                preview,
            )

            try:
                move_to_drop(reachy, target_side, goal_obj)
                get_to_waiting_pose(reachy, duration=3)
            except ValueError:
                print("[loop] Drop pose unreachable — recovering.")
                target_object_unreachable(reachy, target_side)
                time.sleep(3)
                continue

            confirm("Iteration complete. ENTER to start next pick (or Q to stop).")

        except KeyboardInterrupt:
            print("\n[loop] Interrupted by user.")
            break


def shutdown_sequence(reachy: ReachySDK, perception: Optional[Perception] = None) -> None:
    print("\n[shutdown] Beginning safe shutdown...")
    back = DISTANCES["base_back_on_finish"]
    confirm(f"About to back up {back}m and return to default posture.")
    print(f"[shutdown] Backing up {back}m from table...")
    reachy.mobile_base.translate_by(-back, 0, wait=True)
    time.sleep(1)

    print("[shutdown] Returning to default posture...")
    reachy.goto_posture("default", duration=2, wait=True)

    if perception is not None:
        stop_perception(perception)

    print("[shutdown] Turning off smoothly and disconnecting...")
    reachy.turn_off_smoothly()
    reachy.disconnect()
    print("[shutdown] Reachy disconnected. Done.")


def main() -> None:
    reachy: Optional[ReachySDK] = None
    perception: Optional[Perception] = None

    try:
        print("=" * 60)
        print("Reachy the greengrocer")
        print("=" * 60)

        print(f"\n[connect] Connecting to Reachy at {ROBOT_HOST}...")
        reachy = ReachySDK(ROBOT_HOST)
        if not reachy.is_connected:
            raise RuntimeError("Could not connect to Reachy.")
        print(f"[connect] Connected: {reachy.is_connected}")

        confirm("About to power on motors and go to default posture.")
        print("\n[prep] Turning motors on...")
        reachy.turn_on()
        print("[prep] Going to default posture...")
        reachy.goto_posture("default")

        confirm("About to start Perception (loads OwlVit, may take ~1 min).")
        r_cam, perception, owlvit, annotator, labels = build_perception_stack(reachy)

        back = DISTANCES["base_back_to_prep"]
        confirm(f"About to back up {back}m to make space for the arms and come back to origin.")
        print(f"\n[prep] Backing up {back}m to make space for the arms and come back to origin")
        reachy.mobile_base.reset_odometry()
        reachy.mobile_base.goto(-back, 0, 0, wait=True) 
        time.sleep(1)
        get_to_waiting_pose(reachy, duration=3)
        reachy.mobile_base.translate_by(back, 0, wait=True)

        confirm("About to start the live detection loop (continuous feed; ENTER to accept).")
        accepted = live_detection_loop(r_cam, perception, owlvit, annotator, labels)
        print(f"[live] Accepted snapshot ({len(accepted)} tracked objects) — starting manipulation sequence.")

        confirm("About to start the manipulation loop.")
        manipulation_loop(reachy, perception, r_cam, owlvit, annotator, labels)

    except KeyboardInterrupt:
        print("\n[main] Keyboard interrupt received.")
    finally:
        if reachy is not None and reachy.is_connected:
            shutdown_sequence(reachy, perception)


if __name__ == "__main__":
    main()
