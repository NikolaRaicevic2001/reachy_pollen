"""CLI and closed-loop orchestration for reachy_system2."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from reachy_system2.config import (
    ArmReachBand,
    SafeWorkspace,
    max_base_approach_rounds_default,
    perception_snapshot_max_attempts_default,
    reset_odometry_on_run_default,
    settling_s_default,
    verify_max_retries_default,
)
from reachy_system2.executor import ActionExecutor
from reachy_system2.perception import System2Perception, missing_tracked_labels
from reachy_system2.reasoning import ReasoningClient
from reachy_system2.robot_context import (
    format_end_effector_context_for_llm,
    format_scene_hints_for_llm,
    format_workspace_bounds_for_llm,
)
from reachy_system2.run_tracker import RunTracker, log_runs_enabled
from reachy_system2.verify_policy import (
    allows_failure_recovery,
    skip_vision_verification,
    subtask_verification_mode,
)
from reachy_system2.world_frame import (
    OdomSnapshot,
    SceneMemory,
    reset_mobile_base_odometry,
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _connected(reachy) -> bool:
    ic = getattr(reachy, "is_connected", None)
    if ic is None:
        return False
    return ic() if callable(ic) else bool(ic)


_MOBILE_BASE_OPS = frozenset({"mobile_base_translate_by", "mobile_base_rotate_by"})


def _is_mobile_only_subtask(subtask: dict) -> bool:
    """True when every action is a mobile-base op (exploration / base approach)."""
    acts = subtask.get("actions", [])
    if not acts:
        return False
    return all(
        isinstance(a, dict) and a.get("op") in _MOBILE_BASE_OPS for a in acts
    )


_is_exploration_subtask = _is_mobile_only_subtask


def _exploration_target_labels(labels: list[str]) -> list[str]:
    """Labels that must be found before we stop mobile-base exploration.

    Heuristic: ignore obvious supports / containers (table, bowl, plate, counter, surface, desk, shelf).
    This lets exploration focus on the primary manipulandum (e.g. the can), even if the table/bowl
    are only visible from other viewpoints.
    """
    ignore_keywords = ("table", "bowl", "plate", "counter", "surface", "desk", "shelf")
    filtered = []
    for lab in labels:
        name = lab.lower()
        if any(k in name for k in ignore_keywords):
            continue
        filtered.append(lab)
    return filtered or list(labels)


def _place_target_labels(labels: list[str]) -> list[str]:
    """Container / goal labels for the place phase (bowl, plate, etc.)."""
    keywords = ("bowl", "plate", "container", "tray", "bin")
    filtered = []
    for lab in labels:
        name = lab.lower()
        if any(k in name for k in keywords):
            filtered.append(lab)
    return filtered or list(labels)


def _format_odom_for_llm(odom: OdomSnapshot) -> str:
    return (
        f"MOBILE_BASE_ODOMETRY: x={odom.x:.3f}, y={odom.y:.3f}, "
        f"theta_deg={odom.theta_deg:.1f}"
    )


def _format_arm_reach_for_llm(reach: ArmReachBand) -> str:
    eff_xmax = reach.effective_xmax()
    return (
        "ARM_REACH_ZONE (robot frame xy, meters): "
        f"x=[{reach.xmin:.2f}, {eff_xmax:.2f}] (comfort; hard xmax={reach.xmax:.2f}, "
        f"forward margin={reach.forward_x_margin:.2f} m), "
        f"y=[{reach.ymin:.2f}, {reach.ymax:.2f}]. "
        "Targets outside comfort need mobile-base approach before arm motion."
    )


def _generate_manipulation_subtasks(
    *,
    task: str,
    scene: str,
    rgb,
    depth_viz,
    labels: list[str],
    reachy,
    reasoning: ReasoningClient,
    phase: str | None = None,
    memory: SceneMemory | None = None,
    odom: OdomSnapshot | None = None,
) -> list[dict]:
    """Arm subtasks from current perception (pick-only, place-only, or full)."""
    ee_context = format_end_effector_context_for_llm(reachy)
    workspace_bounds = format_workspace_bounds_for_llm(SafeWorkspace.from_env())
    scene_hints = format_scene_hints_for_llm(scene, ee_context)
    world_memory = memory.format_for_llm(odom) if memory is not None and odom is not None else None
    plan = reasoning.generate_plan(
        task,
        scene,
        rgb,
        depth_viz_rgb=depth_viz,
        tracked_labels=labels,
        robot_context=ee_context,
        workspace_bounds=workspace_bounds,
        scene_hints=scene_hints,
        phase=phase,
        world_memory=world_memory,
    )
    return list(plan["subtasks"])


def _execute_subtask_with_oob_replan(
    *,
    subtask: dict,
    executor: ActionExecutor,
    reasoning: ReasoningClient,
    task: str,
    scene: str,
    rgb,
    depth_viz,
    workspace: SafeWorkspace,
    reachy,
    max_oob_replans: int,
    dry_run: bool,
) -> None:
    """Run one subtask; if target xyz is OOB, ask LLM for an in-box replacement."""
    current = subtask
    oob_attempts = 0
    while True:
        validate = executor.validate_subtask_bounds(current)
        if validate.ok:
            break
        if not validate.out_of_workspace:
            raise RuntimeError(validate.message)
        oob_attempts += 1
        if oob_attempts > max_oob_replans:
            raise RuntimeError(
                f"Out-of-workspace replan exceeded limit ({max_oob_replans}). Last: {validate.message}"
            )
        logger.warning("%s — requesting LLM replan (%s/%s)", validate.message, oob_attempts, max_oob_replans)
        bounds = {
            "xmin": workspace.xmin,
            "xmax": workspace.xmax,
            "ymin": workspace.ymin,
            "ymax": workspace.ymax,
            "zmin": workspace.zmin,
            "zmax": workspace.zmax,
        }
        ee_ctx = format_end_effector_context_for_llm(reachy)
        hints = format_scene_hints_for_llm(scene, ee_ctx)
        replan = reasoning.replan_out_of_workspace(
            task=task,
            scene_description=scene,
            rgb=rgb,
            depth_viz_rgb=depth_viz,
            rejected_xyz=validate.rejected_xyz or (0.0, 0.0, 0.0),
            bounds=bounds,
            robot_context=ee_ctx,
            scene_hints=hints,
        )
        if not replan["subtasks"]:
            raise RuntimeError("Replan returned no subtasks.")
        current = replan["subtasks"][0]

    res = executor.run_subtask(current, wait=True, dry_run=dry_run)
    if not res.ok:
        raise RuntimeError(f"Execution failed: {res.message}")


def _failure_suggests_base_nudge(failure_reason: str | None) -> bool:
    if not failure_reason:
        return False
    d = failure_reason.lower()
    keywords = (
        "align",
        "alignment",
        "angle",
        "awkward",
        "reach",
        "stretch",
        "too far",
        "not over",
        "offset",
        "misalign",
    )
    return any(k in d for k in keywords)


def _run_subtask_with_verification_recovery(
    *,
    subtask: dict,
    subtask_index: int,
    total: int,
    task: str,
    labels: list[str],
    pick_target_labels: list[str],
    executor: ActionExecutor,
    reasoning: ReasoningClient,
    perception: System2Perception,
    reachy,
    workspace: SafeWorkspace,
    reach_band: ArmReachBand,
    scene: str,
    rgb,
    depth_viz,
    settling: float,
    detection_threshold: float | None,
    confirm_steps: bool,
    dry_run: bool,
    max_oob_replans: int,
    max_verify_retries: int,
    perception_max_attempts: int,
    tracker: RunTracker | None,
    memory: SceneMemory,
    odom: OdomSnapshot,
    max_base_approach_rounds: int,
) -> tuple[Any, str, Any, list[dict[str, Any]], OdomSnapshot, SceneMemory]:
    """Execute a planned subtask; on verify FAILED, correct or replan from fresh perception."""
    description = str(subtask.get("description", ""))
    verify_mode = subtask_verification_mode(description)
    rgb_before = rgb
    depth_before = depth_viz
    pending_recovery: list[dict] = []
    run_original = True
    recovery_attempt = 0

    while True:
        if run_original:
            if confirm_steps and not dry_run:
                if not confirm_subtask_interactive(
                    subtask,
                    planned_index=subtask_index,
                    planned_total=total,
                    step_kind="planned",
                ):
                    logger.warning("Operator aborted before subtask %s.", subtask_index + 1)
                    raise RuntimeError("Operator aborted subtask")
            _execute_subtask_with_oob_replan(
                subtask=subtask,
                executor=executor,
                reasoning=reasoning,
                task=task,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                workspace=workspace,
                reachy=reachy,
                max_oob_replans=max_oob_replans,
                dry_run=dry_run,
            )
            run_original = False

        recovery_total = len(pending_recovery)
        ran_recovery = 0
        for ri, rs in enumerate(pending_recovery):
            if confirm_steps and not dry_run:
                if not confirm_subtask_interactive(
                    rs,
                    planned_index=subtask_index,
                    planned_total=total,
                    step_kind="recovery",
                    recovery_index=ri,
                    recovery_total=recovery_total,
                ):
                    logger.warning(
                        "Operator skipped recovery step %s/%s for planned subtask %s.",
                        ri + 1,
                        recovery_total,
                        subtask_index + 1,
                    )
                    continue
            _execute_subtask_with_oob_replan(
                subtask=rs,
                executor=executor,
                reasoning=reasoning,
                task=task,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                workspace=workspace,
                reachy=reachy,
                max_oob_replans=max_oob_replans,
                dry_run=dry_run,
            )
            ran_recovery += 1
        if pending_recovery and ran_recovery == 0:
            raise RuntimeError(
                f"Recovery for planned subtask {subtask_index + 1} aborted: all recovery steps skipped."
            )
        pending_recovery = []

        # For mobile-base / other kinematic steps, give OWL-ViT time to update.
        # Otherwise we may re-evaluate "missing" too early right after the viewpoint change.
        if skip_vision_verification(verify_mode):
            snap = perception.snapshot_until_tracked_objects(
                labels=labels,
                detection_threshold=detection_threshold,
                settling_s=settling,
                max_attempts=perception_max_attempts,
                allow_partial=True,
            )
        else:
            snap = perception.snapshot(
                detection_threshold=detection_threshold,
                settling_s=settling,
            )
        rgb_after = snap.rgb
        scene_after = snap.scene
        depth_after = snap.depth_viz_rgb
        if tracker is not None:
            suffix = f"_{recovery_attempt:02d}" if recovery_attempt else ""
            tracker.write_text_file(
                f"perception_after_subtask_{subtask_index:03d}{suffix}.txt",
                scene_after,
            )

        rgb = rgb_after
        scene = scene_after
        depth_viz = depth_after

        if skip_vision_verification(verify_mode):
            logger.info(
                "Skipping vision verification for planned subtask %s (%s): %s",
                subtask_index + 1,
                verify_mode,
                description,
            )
            if tracker is not None:
                tracker.write_json(
                    f"verify_subtask_{subtask_index:03d}.json",
                    {"status": "SKIPPED", "mode": verify_mode, "reason": "kinematic subtask"},
                )
            return rgb, scene, depth_viz, list(snap.objects), odom, memory

        verdict = reasoning.verify_execution(
            goal=task,
            subtask_description=description,
            scene_after=scene_after,
            rgb_after=rgb_after,
            verification_mode=verify_mode,
            rgb_before=rgb_before,
            depth_viz_after=depth_after,
            depth_viz_before=depth_before,
        )
        logger.info(
            "Verification (planned %s/%s, mode=%s, recovery=%s): %s",
            subtask_index + 1,
            total,
            verify_mode,
            recovery_attempt,
            verdict,
        )
        if tracker is not None:
            suffix = f"_{recovery_attempt:02d}" if recovery_attempt else ""
            tracker.write_json(f"verify_subtask_{subtask_index:03d}{suffix}.json", verdict)

        if verdict.get("status") == "OK":
            return rgb, scene, depth_viz, list(snap.objects), odom, memory

        corr = verdict.get("correction")
        if isinstance(corr, dict) and corr.get("actions"):
            run_corr = True
            if confirm_steps and not dry_run:
                run_corr = confirm_subtask_interactive(
                    corr,
                    planned_index=subtask_index,
                    planned_total=total,
                    step_kind="correction",
                )
            if run_corr:
                _execute_subtask_with_oob_replan(
                    subtask=corr,
                    executor=executor,
                    reasoning=reasoning,
                    task=task,
                    scene=scene,
                    rgb=rgb,
                    depth_viz=depth_viz,
                    workspace=workspace,
                    reachy=reachy,
                    max_oob_replans=max_oob_replans,
                    dry_run=dry_run,
                )
                snap_c = perception.snapshot(
                    detection_threshold=detection_threshold,
                    settling_s=settling,
                )
                verdict_c = reasoning.verify_execution(
                    goal=task,
                    subtask_description=description,
                    scene_after=snap_c.scene,
                    rgb_after=snap_c.rgb,
                    verification_mode=verify_mode,
                    rgb_before=rgb_before,
                    depth_viz_after=snap_c.depth_viz_rgb,
                    depth_viz_before=depth_before,
                )
                logger.info("Verification after correction: %s", verdict_c)
                if tracker is not None:
                    tracker.write_json(
                        f"verify_subtask_{subtask_index:03d}_correction.json",
                        verdict_c,
                    )
                rgb, scene, depth_viz = snap_c.rgb, snap_c.scene, snap_c.depth_viz_rgb
                if verdict_c.get("status") == "OK":
                    return rgb, scene, depth_viz, list(snap_c.objects), odom, memory
                verdict = verdict_c

        if not allows_failure_recovery(verify_mode):
            reason = verdict.get("failure_reason") or "unknown"
            raise RuntimeError(
                f"Planned subtask {subtask_index + 1} verification FAILED ({reason!r}) "
                f"but mode {verify_mode!r} does not support recovery replan."
            )

        if recovery_attempt >= max_verify_retries:
            reason = verdict.get("failure_reason") or "unknown"
            raise RuntimeError(
                f"Planned subtask {subtask_index + 1} failed after {max_verify_retries} recovery attempts: "
                f"{description!r} ({reason})"
            )

        recovery_attempt += 1
        failure_reason = verdict.get("failure_reason")
        if (
            verify_mode in ("approach", "grasp")
            and _failure_suggests_base_nudge(failure_reason)
            and recovery_attempt == 1
        ):
            odom = OdomSnapshot.from_reachy(reachy)
            found = memory.find_robot_xyz(
                labels=pick_target_labels,
                detected_objects=list(snap.objects),
                odom=odom,
            )
            if found is not None:
                _, xyz = found
                if not reach_band.contains_xy(xyz[0], xyz[1]):
                    logger.info(
                        "Alignment failure on subtask %s — running base approach before arm recovery "
                        "(target xy=(%.3f, %.3f), reason=%r).",
                        subtask_index + 1,
                        xyz[0],
                        xyz[1],
                        failure_reason,
                    )
                    rgb, scene, depth_viz, objects, odom, memory = (
                        _phase_base_approach_until_reachable(
                            phase_name=f"recovery_subtask_{subtask_index + 1}",
                            target_labels=pick_target_labels,
                            task=task,
                            rgb=rgb,
                            scene=scene,
                            depth_viz=depth_viz,
                            objects=list(snap.objects),
                            memory=memory,
                            odom=odom,
                            reach_band=reach_band,
                            resolved_labels=labels,
                            reasoning=reasoning,
                            executor=executor,
                            perception=perception,
                            reachy=reachy,
                            workspace=workspace,
                            settling=settling,
                            detection_threshold=detection_threshold,
                            confirm_steps=confirm_steps,
                            dry_run=dry_run,
                            max_oob_replans=max_oob_replans,
                            max_verify_retries=max_verify_retries,
                            perception_max_attempts=perception_max_attempts,
                            tracker=tracker,
                            max_rounds=max(1, max_base_approach_rounds),
                        )
                    )
                    snap = perception.snapshot(
                        detection_threshold=detection_threshold,
                        settling_s=settling,
                    )
                    rgb, scene, depth_viz = snap.rgb, snap.scene, snap.depth_viz_rgb
                    odom = OdomSnapshot.from_reachy(reachy)
                    memory.update_from_detections(list(snap.objects), odom)

        ee_ctx = format_end_effector_context_for_llm(reachy)
        hints = format_scene_hints_for_llm(scene, ee_ctx)
        bounds_txt = format_workspace_bounds_for_llm(workspace)
        odom = OdomSnapshot.from_reachy(reachy)
        replan = reasoning.replan_after_failure(
            task=task,
            failed_subtask_description=description,
            failed_subtask=subtask,
            verification=verdict,
            scene_description=scene,
            rgb=rgb,
            robot_context=ee_ctx,
            workspace_bounds=bounds_txt,
            scene_hints=hints,
            depth_viz_rgb=depth_viz,
            tracked_labels=labels,
            world_memory=memory.format_for_llm(odom),
            mobile_base_odometry=_format_odom_for_llm(odom),
        )
        logger.info(
            "Recovery replan (%s/%s): %s",
            recovery_attempt,
            max_verify_retries,
            json.dumps(replan, indent=2)[:4000],
        )
        if tracker is not None:
            tracker.write_json(
                f"replan_failure_subtask_{subtask_index:03d}_{recovery_attempt:02d}.json",
                replan,
            )
        pending_recovery = list(replan["subtasks"])
        if not pending_recovery:
            raise RuntimeError("Recovery replan returned no subtasks.")


def confirm_subtask_interactive(
    subtask: dict,
    *,
    planned_index: int,
    planned_total: int,
    step_kind: str = "planned",
    recovery_index: int | None = None,
    recovery_total: int | None = None,
) -> bool:
    if step_kind == "planned":
        label = f"Planned subtask {planned_index + 1}/{planned_total}"
    elif step_kind == "correction":
        label = f"Correction (after planned subtask {planned_index + 1}/{planned_total})"
    elif step_kind == "recovery":
        r_i = (recovery_index or 0) + 1
        r_t = recovery_total or 1
        label = (
            f"Recovery step {r_i}/{r_t} "
            f"(for planned subtask {planned_index + 1}/{planned_total})"
        )
    else:
        label = f"Step (planned subtask {planned_index + 1}/{planned_total})"
    print(f"\n--- {label} ---")
    print(json.dumps(subtask, indent=2))
    # In Jupyter/Cursor, input() can look like a hang if the prompt is easy to miss; flush first.
    print(
        "\n>>> Waiting for confirmation: type y + Enter to run this on the robot, "
        "or n + Enter to abort. <<<",
        flush=True,
    )
    ans = input("Execute on hardware? [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def _run_subtask_queue(
    subs: list[dict],
    *,
    start_index: int,
    task: str,
    labels: list[str],
    pick_target_labels: list[str],
    mobile_snapshot_labels: list[str],
    executor: ActionExecutor,
    reasoning: ReasoningClient,
    perception: System2Perception,
    reachy,
    workspace: SafeWorkspace,
    reach_band: ArmReachBand,
    scene: str,
    rgb,
    depth_viz,
    settling: float,
    detection_threshold: float | None,
    confirm_steps: bool,
    dry_run: bool,
    max_oob_replans: int,
    max_verify_retries: int,
    perception_max_attempts: int,
    max_base_approach_rounds: int,
    tracker: RunTracker | None,
    memory: SceneMemory,
    odom: OdomSnapshot,
) -> tuple[Any, str, Any, list[dict[str, Any]], OdomSnapshot, SceneMemory, int]:
    """Execute subs[start_index:] with verification; refresh world memory after each step."""
    i = start_index
    objects: list[dict[str, Any]] = []
    while i < len(subs):
        sub = subs[i]
        mobile = _is_mobile_only_subtask(sub)
        snap_labels = mobile_snapshot_labels if mobile else labels
        logger.info(
            "Next subtask %s/%s (%s): %s",
            i + 1,
            len(subs),
            "mobile" if mobile else "manipulation",
            sub.get("description", ""),
        )
        try:
            rgb, scene, depth_viz, objects, odom, memory = _run_subtask_with_verification_recovery(
                subtask=sub,
                subtask_index=i,
                total=len(subs),
                task=task,
                labels=snap_labels,
                pick_target_labels=pick_target_labels,
                executor=executor,
                reasoning=reasoning,
                perception=perception,
                reachy=reachy,
                workspace=workspace,
                reach_band=reach_band,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                settling=settling,
                detection_threshold=detection_threshold,
                confirm_steps=confirm_steps,
                dry_run=dry_run,
                max_oob_replans=max_oob_replans,
                max_verify_retries=max_verify_retries,
                perception_max_attempts=perception_max_attempts,
                tracker=tracker,
                memory=memory,
                odom=odom,
                max_base_approach_rounds=max_base_approach_rounds,
            )
        except RuntimeError as e:
            if "Operator aborted" in str(e):
                raise
            raise
        odom = OdomSnapshot.from_reachy(reachy)
        memory.update_from_detections(objects, odom)
        i += 1
    return rgb, scene, depth_viz, objects, odom, memory, i


def _phase_explore_until_visible(
    *,
    task: str,
    exploration_labels: list[str],
    resolved_labels: list[str],
    rgb,
    scene: str,
    depth_viz,
    objects: list[dict[str, Any]],
    memory: SceneMemory,
    odom: OdomSnapshot,
    reasoning: ReasoningClient,
    executor: ActionExecutor,
    perception: System2Perception,
    reachy,
    workspace: SafeWorkspace,
    reach_band: ArmReachBand,
    settling: float,
    detection_threshold: float | None,
    confirm_steps: bool,
    dry_run: bool,
    max_oob_replans: int,
    max_verify_retries: int,
    perception_max_attempts: int,
    max_base_approach_rounds: int,
    tracker: RunTracker | None,
    max_explore_rounds: int = 3,
) -> tuple[Any, str, Any, list[dict[str, Any]], OdomSnapshot, SceneMemory]:
    missing = missing_tracked_labels(objects, exploration_labels)
    if not missing:
        logger.info("Exploration skipped: pick target already visible (%s).", exploration_labels)
        return rgb, scene, depth_viz, objects, odom, memory

    explore_rounds = 0
    while missing:
        explore_rounds += 1
        if explore_rounds > max_explore_rounds:
            raise RuntimeError(
                f"Could not find pick target after {max_explore_rounds} exploration rounds. "
                f"Still missing: {missing!r}"
            )
        logger.warning(
            "Exploration round %s/%s — missing pick target: %s",
            explore_rounds,
            max_explore_rounds,
            missing,
        )
        if tracker is not None:
            tracker.write_json(
                f"missing_labels_round_{explore_rounds:02d}.json",
                {"missing": missing},
            )
        explore = reasoning.generate_exploration_plan(
            task=task,
            scene_description=scene,
            rgb=rgb,
            depth_viz_rgb=depth_viz,
            missing_labels=missing,
        )
        subs = list(explore.get("subtasks", []))
        if not subs:
            raise RuntimeError("Exploration plan returned no subtasks.")
        if tracker is not None:
            tracker.write_json(f"explore_plan_round_{explore_rounds:02d}.json", {"subtasks": subs})
        rgb, scene, depth_viz, objects, odom, memory, _ = _run_subtask_queue(
            subs,
            start_index=0,
            task=task,
            labels=resolved_labels,
            pick_target_labels=exploration_labels,
            mobile_snapshot_labels=exploration_labels,
            executor=executor,
            reasoning=reasoning,
            perception=perception,
            reachy=reachy,
            workspace=workspace,
            reach_band=reach_band,
            scene=scene,
            rgb=rgb,
            depth_viz=depth_viz,
            settling=settling,
            detection_threshold=detection_threshold,
            confirm_steps=confirm_steps,
            dry_run=dry_run,
            max_oob_replans=max_oob_replans,
            max_verify_retries=max_verify_retries,
            perception_max_attempts=perception_max_attempts,
            max_base_approach_rounds=max_base_approach_rounds,
            tracker=tracker,
            memory=memory,
            odom=odom,
        )
        missing = missing_tracked_labels(objects, exploration_labels)
    return rgb, scene, depth_viz, objects, odom, memory


def _phase_base_approach_until_reachable(
    *,
    phase_name: str,
    target_labels: list[str],
    task: str,
    rgb,
    scene: str,
    depth_viz,
    objects: list[dict[str, Any]],
    memory: SceneMemory,
    odom: OdomSnapshot,
    reach_band: ArmReachBand,
    resolved_labels: list[str],
    reasoning: ReasoningClient,
    executor: ActionExecutor,
    perception: System2Perception,
    reachy,
    workspace: SafeWorkspace,
    settling: float,
    detection_threshold: float | None,
    confirm_steps: bool,
    dry_run: bool,
    max_oob_replans: int,
    max_verify_retries: int,
    perception_max_attempts: int,
    max_base_approach_rounds: int,
    tracker: RunTracker | None,
    max_rounds: int,
) -> tuple[Any, str, Any, list[dict[str, Any]], OdomSnapshot, SceneMemory]:
    arm_reach_txt = _format_arm_reach_for_llm(reach_band)
    for round_idx in range(1, max_rounds + 1):
        odom = OdomSnapshot.from_reachy(reachy)
        found = memory.find_robot_xyz(
            labels=target_labels,
            detected_objects=objects,
            odom=odom,
        )
        if found is None:
            logger.warning(
                "%s base approach: no xyz for labels %s (live or world memory).",
                phase_name,
                target_labels,
            )
            return rgb, scene, depth_viz, objects, odom, memory

        name, xyz = found
        if reach_band.contains_xy(xyz[0], xyz[1]):
            logger.info(
                "%s: target %r at xy=(%.3f, %.3f) is within arm reach.",
                phase_name,
                name,
                xyz[0],
                xyz[1],
            )
            return rgb, scene, depth_viz, objects, odom, memory

        logger.info(
            "%s base approach round %s/%s: %r at xyz=%s outside arm reach — planning mobile moves.",
            phase_name,
            round_idx,
            max_rounds,
            name,
            [round(v, 3) for v in xyz],
        )
        approach = reasoning.generate_base_approach_plan(
            task=task,
            target_label=name,
            target_xyz=xyz,
            arm_reach_zone=arm_reach_txt,
            scene_description=scene,
            rgb=rgb,
            world_memory=memory.format_for_llm(odom),
            mobile_base_odometry=_format_odom_for_llm(odom),
            depth_viz_rgb=depth_viz,
        )
        subs = list(approach.get("subtasks", []))
        if not subs:
            logger.warning("%s base approach plan empty.", phase_name)
            return rgb, scene, depth_viz, objects, odom, memory
        if tracker is not None:
            tracker.write_json(
                f"base_approach_{phase_name}_round_{round_idx:02d}.json",
                {"subtasks": subs, "target": name, "xyz": list(xyz)},
            )
        rgb, scene, depth_viz, objects, odom, memory, _ = _run_subtask_queue(
            subs,
            start_index=0,
            task=task,
            labels=resolved_labels,
            pick_target_labels=target_labels,
            mobile_snapshot_labels=target_labels,
            executor=executor,
            reasoning=reasoning,
            perception=perception,
            reachy=reachy,
            workspace=workspace,
            reach_band=reach_band,
            scene=scene,
            rgb=rgb,
            depth_viz=depth_viz,
            settling=settling,
            detection_threshold=detection_threshold,
            confirm_steps=confirm_steps,
            dry_run=dry_run,
            max_oob_replans=max_oob_replans,
            max_verify_retries=max_verify_retries,
            perception_max_attempts=perception_max_attempts,
            max_base_approach_rounds=max_base_approach_rounds,
            tracker=tracker,
            memory=memory,
            odom=odom,
        )
    logger.warning("%s: max base approach rounds (%s) reached.", phase_name, max_rounds)
    return rgb, scene, depth_viz, objects, odom, memory


def run_closed_loop(
    *,
    reachy,
    task: str,
    labels: list[str] | None = None,
    detection_threshold: float | None = None,
    settling_s: float | None = None,
    confirm_steps: bool = True,
    dry_run: bool = False,
    model: str | None = None,
    max_oob_replans: int = 3,
    max_verify_retries: int | None = None,
    log_runs: bool | None = None,
    run_tracker: RunTracker | None = None,
    robot_host: str | None = None,
    perception_max_attempts: int | None = None,
) -> None:
    """Run perception → plan → execute → verify loop (used by CLI and notebook)."""
    settling = float(settling_s if settling_s is not None else settling_s_default())
    verify_retries = (
        int(max_verify_retries)
        if max_verify_retries is not None
        else verify_max_retries_default()
    )
    workspace = SafeWorkspace.from_env()
    reach_band = ArmReachBand.from_env()
    max_base_rounds = max_base_approach_rounds_default()
    do_log = log_runs if log_runs is not None else log_runs_enabled()
    tracker = run_tracker
    if do_log and tracker is None:
        tracker = RunTracker()

    reasoning = ReasoningClient(model=model, run_tracker=tracker)
    executor = ActionExecutor(reachy, workspace=workspace)
    perception = System2Perception(reachy, detection_threshold=detection_threshold)

    run_status = "ok"
    run_error: str | None = None

    resolved_labels = [s.strip() for s in (labels or []) if s.strip()]
    labels_from_env = bool(resolved_labels)
    if not resolved_labels:
        logger.info("No labels provided — inferring from task via LLM.")
        resolved_labels = reasoning.infer_tracked_labels(task)
        if tracker is not None:
            tracker.write_json(
                "labels.json",
                {"labels": resolved_labels, "source": "llm"},
            )

    exploration_labels = _exploration_target_labels(resolved_labels)
    place_labels = _place_target_labels(resolved_labels)

    if reset_odometry_on_run_default():
        reset_mobile_base_odometry(reachy)
    memory = SceneMemory()
    odom = OdomSnapshot.from_reachy(reachy)

    perception.set_tracked_labels(resolved_labels)
    perception.start(visualize=False)
    try:
        if tracker is not None:
            tracker.write_run_meta(
                {
                    "task": task,
                    "labels": resolved_labels,
                    "labels_source": "env" if labels_from_env else "llm",
                    "model": reasoning.model,
                    "dry_run": dry_run,
                    "confirm_steps": confirm_steps,
                    "settling_s": settling,
                    "detection_threshold": detection_threshold,
                    "robot_host": robot_host,
                    "perception_max_attempts": perception_max_attempts
                    if perception_max_attempts is not None
                    else perception_snapshot_max_attempts_default(),
                    "max_verify_retries": verify_retries,
                    "arm_reach_band": {
                        "xmin": reach_band.xmin,
                        "xmax": reach_band.xmax,
                        "effective_xmax": reach_band.effective_xmax(),
                        "forward_x_margin": reach_band.forward_x_margin,
                        "ymin": reach_band.ymin,
                        "ymax": reach_band.ymax,
                    },
                    "max_base_approach_rounds": max_base_rounds,
                    "reset_odometry": reset_odometry_on_run_default(),
                }
            )

        max_att = (
            int(perception_max_attempts)
            if perception_max_attempts is not None
            else perception_snapshot_max_attempts_default()
        )

        # Initial perception (partial ok — missing pick target triggers exploration first).
        snap = perception.snapshot_until_tracked_objects(
            labels=resolved_labels,
            detection_threshold=detection_threshold,
            settling_s=settling,
            max_attempts=max_att,
            allow_partial=True,
        )
        rgb, scene, depth_viz = snap.rgb, snap.scene, snap.depth_viz_rgb
        objects = list(snap.objects)
        memory.update_from_detections(objects, odom)
        if tracker is not None:
            tracker.write_text_file("perception_initial.txt", scene)
            tracker.write_text_file("world_memory_initial.txt", memory.format_for_llm(odom))

        ee_context = format_end_effector_context_for_llm(reachy)
        workspace_bounds = format_workspace_bounds_for_llm(workspace)
        scene_hints = format_scene_hints_for_llm(scene, ee_context)
        if tracker is not None:
            tracker.write_text_file("robot_context_initial.txt", ee_context)
            tracker.write_text_file("workspace_bounds.txt", workspace_bounds)
            tracker.write_text_file("scene_hints.txt", scene_hints)

        queue_kwargs = dict(
            task=task,
            executor=executor,
            reasoning=reasoning,
            perception=perception,
            reachy=reachy,
            workspace=workspace,
            reach_band=reach_band,
            settling=settling,
            detection_threshold=detection_threshold,
            confirm_steps=confirm_steps,
            dry_run=dry_run,
            max_oob_replans=max_oob_replans,
            max_verify_retries=verify_retries,
            perception_max_attempts=max_att,
            max_base_approach_rounds=max_base_rounds,
            tracker=tracker,
            memory=memory,
            odom=odom,
        )

        # Phased pick → place with world memory and base approach.
        try:
            logger.info("=== Phase 1: explore for pick target %s ===", exploration_labels)
            rgb, scene, depth_viz, objects, odom, memory = _phase_explore_until_visible(
                exploration_labels=exploration_labels,
                resolved_labels=resolved_labels,
                rgb=rgb,
                scene=scene,
                depth_viz=depth_viz,
                objects=objects,
                **queue_kwargs,
            )

            logger.info("=== Phase 2: base approach to pick target ===")
            rgb, scene, depth_viz, objects, odom, memory = _phase_base_approach_until_reachable(
                phase_name="pick",
                target_labels=exploration_labels,
                rgb=rgb,
                scene=scene,
                depth_viz=depth_viz,
                objects=objects,
                resolved_labels=resolved_labels,
                max_rounds=max_base_rounds,
                **queue_kwargs,
            )

            logger.info("=== Phase 3: pick (arm) ===")
            pick_subs = _generate_manipulation_subtasks(
                task=task,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                labels=resolved_labels,
                reachy=reachy,
                reasoning=reasoning,
                phase="pick",
                memory=memory,
                odom=odom,
            )
            if tracker is not None:
                tracker.write_json("plan_pick.json", {"subtasks": pick_subs})
            logger.info("Pick plan (%s subtasks)", len(pick_subs))
            rgb, scene, depth_viz, objects, odom, memory, _ = _run_subtask_queue(
                pick_subs,
                start_index=0,
                labels=resolved_labels,
                pick_target_labels=exploration_labels,
                mobile_snapshot_labels=exploration_labels,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                **queue_kwargs,
            )

            logger.info("=== Phase 4: base approach to place target %s ===", place_labels)
            rgb, scene, depth_viz, objects, odom, memory = _phase_base_approach_until_reachable(
                phase_name="place",
                target_labels=place_labels,
                rgb=rgb,
                scene=scene,
                depth_viz=depth_viz,
                objects=objects,
                resolved_labels=resolved_labels,
                max_rounds=max_base_rounds,
                **queue_kwargs,
            )

            logger.info("=== Phase 5: place (arm) ===")
            place_subs = _generate_manipulation_subtasks(
                task=task,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                labels=resolved_labels,
                reachy=reachy,
                reasoning=reasoning,
                phase="place",
                memory=memory,
                odom=odom,
            )
            if tracker is not None:
                tracker.write_json("plan_place.json", {"subtasks": place_subs})
            logger.info("Place plan (%s subtasks)", len(place_subs))
            rgb, scene, depth_viz, objects, odom, memory, _ = _run_subtask_queue(
                place_subs,
                start_index=0,
                labels=resolved_labels,
                pick_target_labels=place_labels,
                mobile_snapshot_labels=place_labels,
                scene=scene,
                rgb=rgb,
                depth_viz=depth_viz,
                **queue_kwargs,
            )

            logger.info("=== Task complete ===")
            if tracker is not None:
                tracker.write_text_file("world_memory_final.txt", memory.format_for_llm(odom))
        except RuntimeError as e:
            if "Operator aborted" in str(e):
                run_status = "aborted"
                return
            raise
    except Exception as e:
        run_status = "error"
        run_error = str(e)
        raise
    finally:
        perception.stop()
        if tracker is not None:
            tracker.finalize(status=run_status, error=run_error)
            logger.info("Run log written under %s", tracker.path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(_repo_root() / ".env")

    parser = argparse.ArgumentParser(description="Reachy System 2 closed-loop (perception + LLM + executor).")
    parser.add_argument("--host", default=os.environ.get("REACHY_HOST"), help="Robot host / IP (or REACHY_HOST).")
    parser.add_argument("--task", default=os.environ.get("SYSTEM2_TASK"), required=False, help="High-level goal.")
    parser.add_argument(
        "--labels",
        default=os.environ.get("SYSTEM2_LABELS"),
        help="Optional comma-separated labels; if omitted, LLM infers labels from --task first.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"), help="OpenAI model name.")
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=None,
        help="Override Perception get_objects_infos threshold.",
    )
    parser.add_argument(
        "--settling-s",
        type=float,
        default=None,
        help="Seconds to wait before each snapshot (vibration / vision settling).",
    )
    parser.add_argument(
        "--no-confirm-steps",
        dest="confirm_steps",
        action="store_false",
        help="Do not prompt before each subtask (unattended).",
    )
    parser.set_defaults(confirm_steps=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and skip real robot motion (still calls LLM if keys set).",
    )
    parser.add_argument(
        "--no-log-runs",
        action="store_true",
        help="Disable run directory and token logging (see also SYSTEM2_LOG_RUNS in .env).",
    )
    parser.add_argument(
        "--perception-max-attempts",
        type=int,
        default=None,
        help="Retries before planning until detections match labels (or SYSTEM2_SNAPSHOT_MAX_ATTEMPTS).",
    )
    parser.add_argument(
        "--max-verify-retries",
        type=int,
        default=None,
        help="Recovery replans per subtask after FAILED verification (or SYSTEM2_MAX_VERIFY_RETRIES).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.host:
        logger.error("Missing robot host: pass --host or set REACHY_HOST in .env")
        return 2
    if not args.task:
        logger.error("Missing --task or SYSTEM2_TASK in .env")
        return 2
    from reachy2_sdk import ReachySDK

    reachy = ReachySDK(args.host)
    if not _connected(reachy):
        logger.error("Could not connect to Reachy at %s", args.host)
        return 3

    labels = (
        [s.strip() for s in args.labels.split(",") if s.strip()]
        if args.labels and args.labels.strip()
        else None
    )
    log_runs = log_runs_enabled() and not args.no_log_runs
    run_closed_loop(
        reachy=reachy,
        task=args.task,
        labels=labels,
        detection_threshold=args.detection_threshold,
        settling_s=args.settling_s,
        confirm_steps=args.confirm_steps,
        dry_run=args.dry_run,
        model=args.model,
        log_runs=log_runs,
        robot_host=args.host,
        perception_max_attempts=args.perception_max_attempts,
        max_verify_retries=args.max_verify_retries,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
