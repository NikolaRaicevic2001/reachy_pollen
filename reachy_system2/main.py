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
    SafeWorkspace,
    perception_snapshot_max_attempts_default,
    settling_s_default,
    verify_max_retries_default,
)
from reachy_system2.executor import ActionExecutor
from reachy_system2.perception import System2Perception
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

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _connected(reachy) -> bool:
    ic = getattr(reachy, "is_connected", None)
    if ic is None:
        return False
    return ic() if callable(ic) else bool(ic)


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


def _run_subtask_with_verification_recovery(
    *,
    subtask: dict,
    subtask_index: int,
    total: int,
    task: str,
    labels: list[str],
    executor: ActionExecutor,
    reasoning: ReasoningClient,
    perception: System2Perception,
    reachy,
    workspace: SafeWorkspace,
    scene: str,
    rgb,
    depth_viz,
    settling: float,
    detection_threshold: float | None,
    confirm_steps: bool,
    dry_run: bool,
    max_oob_replans: int,
    max_verify_retries: int,
    tracker: RunTracker | None,
) -> tuple[Any, str, Any]:
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
            return rgb, scene, depth_viz

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
            return rgb, scene, depth_viz

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
                    return rgb, scene, depth_viz
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
        ee_ctx = format_end_effector_context_for_llm(reachy)
        hints = format_scene_hints_for_llm(scene, ee_ctx)
        bounds_txt = format_workspace_bounds_for_llm(workspace)
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
                }
            )

        max_att = (
            int(perception_max_attempts)
            if perception_max_attempts is not None
            else perception_snapshot_max_attempts_default()
        )

        snap = perception.snapshot_until_tracked_objects(
            labels=resolved_labels,
            detection_threshold=detection_threshold,
            settling_s=settling,
            max_attempts=max_att,
        )
        rgb, scene, depth_viz = snap.rgb, snap.scene, snap.depth_viz_rgb
        if tracker is not None:
            tracker.write_text_file("perception_initial.txt", scene)

        ee_context = format_end_effector_context_for_llm(reachy)
        workspace_bounds = format_workspace_bounds_for_llm(workspace)
        scene_hints = format_scene_hints_for_llm(scene, ee_context)
        if tracker is not None:
            tracker.write_text_file("robot_context_initial.txt", ee_context)
            tracker.write_text_file("workspace_bounds.txt", workspace_bounds)
            tracker.write_text_file("scene_hints.txt", scene_hints)

        plan = reasoning.generate_plan(
            task,
            scene,
            rgb,
            depth_viz_rgb=depth_viz,
            tracked_labels=resolved_labels,
            robot_context=ee_context,
            workspace_bounds=workspace_bounds,
            scene_hints=scene_hints,
        )
        logger.info("Plan: %s", json.dumps(plan, indent=2)[:8000])
        if tracker is not None:
            tracker.write_json("plan.json", plan)

        subs = plan["subtasks"]
        total = len(subs)
        for i, sub in enumerate(subs):
            try:
                rgb, scene, depth_viz = _run_subtask_with_verification_recovery(
                    subtask=sub,
                    subtask_index=i,
                    total=total,
                    task=task,
                    labels=resolved_labels,
                    executor=executor,
                    reasoning=reasoning,
                    perception=perception,
                    reachy=reachy,
                    workspace=workspace,
                    scene=scene,
                    rgb=rgb,
                    depth_viz=depth_viz,
                    settling=settling,
                    detection_threshold=detection_threshold,
                    confirm_steps=confirm_steps,
                    dry_run=dry_run,
                    max_oob_replans=max_oob_replans,
                    max_verify_retries=verify_retries,
                    tracker=tracker,
                )
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
