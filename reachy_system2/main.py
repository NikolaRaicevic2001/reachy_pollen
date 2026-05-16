"""CLI and closed-loop orchestration for reachy_system2."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from reachy_system2.config import (
    SafeWorkspace,
    perception_snapshot_max_attempts_default,
    require_every_tracked_label_default,
    settling_s_default,
)
from reachy_system2.executor import ActionExecutor
from reachy_system2.perception import System2Perception
from reachy_system2.reasoning import ReasoningClient
from reachy_system2.robot_context import format_end_effector_context_for_llm
from reachy_system2.run_tracker import RunTracker, log_runs_enabled

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _connected(reachy) -> bool:
    ic = getattr(reachy, "is_connected", None)
    if ic is None:
        return False
    return ic() if callable(ic) else bool(ic)


def confirm_subtask_interactive(
    subtask: dict,
    *,
    index: int,
    total: int,
    correction: bool = False,
) -> bool:
    label = "Correction subtask" if correction else "Subtask"
    print(f"\n--- {label} {index + 1}/{total} ---")
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
    labels: list[str],
    detection_threshold: float | None = None,
    settling_s: float | None = None,
    confirm_steps: bool = True,
    dry_run: bool = False,
    model: str | None = None,
    max_oob_replans: int = 3,
    log_runs: bool | None = None,
    run_tracker: RunTracker | None = None,
    robot_host: str | None = None,
    perception_max_attempts: int | None = None,
    require_every_tracked_label: bool | None = None,
) -> None:
    """Run perception → plan → execute → verify loop (used by CLI and notebook)."""
    settling = float(settling_s if settling_s is not None else settling_s_default())
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

    perception.set_tracked_labels(labels)
    perception.start(visualize=False)
    try:
        if tracker is not None:
            tracker.write_run_meta(
                {
                    "task": task,
                    "labels": labels,
                    "model": reasoning.model,
                    "dry_run": dry_run,
                    "confirm_steps": confirm_steps,
                    "settling_s": settling,
                    "detection_threshold": detection_threshold,
                    "robot_host": robot_host,
                    "perception_max_attempts": perception_max_attempts
                    if perception_max_attempts is not None
                    else perception_snapshot_max_attempts_default(),
                    "require_every_tracked_label": require_every_tracked_label
                    if require_every_tracked_label is not None
                    else require_every_tracked_label_default(),
                }
            )

        max_att = (
            int(perception_max_attempts)
            if perception_max_attempts is not None
            else perception_snapshot_max_attempts_default()
        )
        req_all_labels = (
            require_every_tracked_label
            if require_every_tracked_label is not None
            else require_every_tracked_label_default()
        )

        snap = perception.snapshot_until_tracked_objects(
            labels=labels,
            detection_threshold=detection_threshold,
            settling_s=settling,
            max_attempts=max_att,
            require_every_label=req_all_labels,
        )
        rgb, scene = snap.rgb, snap.scene
        if tracker is not None:
            tracker.write_text_file("perception_initial.txt", scene)

        ee_context = format_end_effector_context_for_llm(reachy)
        if tracker is not None:
            tracker.write_text_file("robot_context_initial.txt", ee_context)

        plan = reasoning.generate_plan(
            task,
            scene,
            rgb,
            tracked_labels=labels,
            robot_context=ee_context,
        )
        logger.info("Plan: %s", json.dumps(plan, indent=2)[:8000])
        if tracker is not None:
            tracker.write_json("plan.json", plan)

        subs = plan["subtasks"]
        total = len(subs)
        for i, sub in enumerate(subs):
            if confirm_steps and not dry_run:
                if not confirm_subtask_interactive(sub, index=i, total=total):
                    logger.warning("Operator aborted before subtask %s.", i + 1)
                    run_status = "aborted"
                    return

            oob_attempts = 0
            current = sub
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
                replan = reasoning.replan_out_of_workspace(
                    task=task,
                    scene_description=scene,
                    rgb=rgb,
                    rejected_xyz=validate.rejected_xyz or (0.0, 0.0, 0.0),
                    bounds=bounds,
                    robot_context=ee_ctx,
                )
                if not replan["subtasks"]:
                    raise RuntimeError("Replan returned no subtasks.")
                current = replan["subtasks"][0]

            res = executor.run_subtask(current, wait=True, dry_run=dry_run)
            if not res.ok:
                raise RuntimeError(f"Execution failed: {res.message}")

            snap_after = perception.snapshot(
                detection_threshold=detection_threshold,
                settling_s=settling,
            )
            rgb_after, scene_after = snap_after.rgb, snap_after.scene
            verdict = reasoning.verify_execution(
                goal=task,
                subtask_description=str(current.get("description", "")),
                scene_after=scene_after,
                rgb_after=rgb_after,
                rgb_before=rgb,
            )
            logger.info("Verification: %s", verdict)
            if tracker is not None:
                tracker.write_json(f"verify_subtask_{i:03d}.json", verdict)
            rgb = rgb_after
            scene = scene_after

            if verdict.get("status") != "OK":
                corr = verdict.get("correction")
                if isinstance(corr, dict) and corr.get("actions"):
                    if confirm_steps and not dry_run:
                        if not confirm_subtask_interactive(
                            corr, index=i, total=total, correction=True
                        ):
                            logger.warning("Operator skipped correction for subtask %s.", i + 1)
                            continue
                    cres = executor.run_subtask(corr, wait=True, dry_run=dry_run)
                    if not cres.ok:
                        raise RuntimeError(f"Correction failed: {cres.message}")
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
        help="Comma-separated object labels for Perception (e.g. bowl,apple).",
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
        "--any-label-match",
        dest="require_every_tracked_label",
        action="store_false",
        help="Stop waiting when at least one object is detected (not every label).",
    )
    parser.set_defaults(require_every_tracked_label=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.host:
        logger.error("Missing robot host: pass --host or set REACHY_HOST in .env")
        return 2
    if not args.task:
        logger.error("Missing --task or SYSTEM2_TASK in .env")
        return 2
    if not args.labels:
        logger.error("Missing --labels or SYSTEM2_LABELS in .env")
        return 2

    from reachy2_sdk import ReachySDK

    reachy = ReachySDK(args.host)
    if not _connected(reachy):
        logger.error("Could not connect to Reachy at %s", args.host)
        return 3

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
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
        require_every_tracked_label=args.require_every_tracked_label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
