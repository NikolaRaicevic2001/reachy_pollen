"""OpenAI vision client: planning, verification, out-of-workspace replanning."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Sequence

import numpy as np
from openai import OpenAI

from reachy_system2.prompts import (
    LABELS_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    REPLAN_AFTER_FAILURE_SYSTEM_PROMPT,
    REPLAN_OOB_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)
from reachy_system2.run_tracker import RunTracker

logger = logging.getLogger(__name__)

_MAX_ACTIONS = 5
_MAX_SUBTASKS = 12

_ALLOWED_ACTION_OPS = frozenset(
    {
        "r_arm_goto_pose",
        "l_arm_goto_pose",
        "r_arm_translate",
        "l_arm_translate",
        "r_arm_rotate",
        "l_arm_rotate",
        "r_gripper_goto",
        "l_gripper_goto",
    }
)

# Shorthand ops models emit; mapped to r_* by default (override with action["arm"]="l").
_OP_SHORTHAND_TO_SUFFIX: dict[str, str] = {
    "goto_pose": "arm_goto_pose",
    "arm_goto_pose": "arm_goto_pose",
    "translate": "arm_translate",
    "rotate": "arm_rotate",
    "gripper_goto": "gripper_goto",
    "gripper": "gripper_goto",
}


def _arm_prefix_from_action(action: dict[str, Any]) -> str:
    hint = action.get("arm") or action.get("side") or action.get("arm_name")
    if hint is None:
        return "r"
    s = str(hint).strip().lower()
    if s in ("l", "left", "l_arm", "l_arm_goto_pose"):
        return "l"
    return "r"


def _canonicalize_op_name(op: str, action: dict[str, Any]) -> str:
    """Map generic op names (e.g. ``goto_pose``) to ``r_arm_goto_pose`` / ``l_arm_goto_pose``."""
    if op in _ALLOWED_ACTION_OPS:
        return op
    suffix = _OP_SHORTHAND_TO_SUFFIX.get(op)
    if suffix is not None:
        prefix = _arm_prefix_from_action(action)
        canonical = f"{prefix}_{suffix}"
        if canonical in _ALLOWED_ACTION_OPS:
            logger.debug("Mapped op %r -> %r", op, canonical)
            return canonical
    return op


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _fence_body_anywhere(text: str) -> str | None:
    """First ``` or ```json ... ``` block anywhere in the reply (models often add prose around it)."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _first_balanced_json_object(text: str) -> str | None:
    """Return substring of first top-level `{` … `}` pair (handles preamble before JSON)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_llm_json(raw: str, *, context: str) -> dict[str, Any]:
    """Parse JSON from model text; tolerate fences and leading/trailing commentary."""
    raw = raw or ""
    stripped = _strip_json_fence(raw.strip())
    if not stripped and raw.strip():
        stripped = raw.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    fb = _fence_body_anywhere(raw)
    if fb and fb not in candidates:
        candidates.append(fb)
    obj = _first_balanced_json_object(raw)
    if obj and obj not in candidates:
        candidates.append(obj)

    last_err: Exception | None = None
    for s in candidates:
        if not s:
            continue
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as e:
            last_err = e
            continue

    preview = (raw[:800] + "…") if len(raw) > 800 else raw
    if not raw.strip():
        raise ValueError(
            f"{context}: model returned empty message content (check API key, model name, quota, or refusal)."
        ) from last_err
    raise ValueError(
        f"{context}: could not parse JSON from model reply. First 800 chars:\n{preview!r}"
    ) from last_err


def _encode_image_jpeg_b64(rgb: np.ndarray) -> str:
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
    bgr = rgb[:, :, :3]
    if rgb.shape[2] == 4:
        bgr = bgr[:, :, :3]
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPEG")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


_DEPTH_IMAGE_CAPTION = (
    "DEPTH_IMAGE (torso camera, TURBO colormap aligned to RGB: warm=nearer, cool=farther, "
    "black=no reading). Use with PERCEPTION for table height, surfaces, and approach clearance."
)


def _user_multimodal_rgb_depth(
    text: str,
    rgb: np.ndarray,
    depth_viz_rgb: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.append({"type": "text", "text": "RGB_IMAGE (torso camera):"})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_jpeg_b64(rgb)}"},
        },
    )
    if depth_viz_rgb is not None:
        content.append({"type": "text", "text": _DEPTH_IMAGE_CAPTION})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_jpeg_b64(depth_viz_rgb)}"},
            },
        )
    return content


def _normalize_action_dict(action: Any, *, context: str, index: int) -> dict[str, Any]:
    """Flatten common LLM variants to executor shape: ``{"op": "...", ...}``."""
    if not isinstance(action, dict):
        raise ValueError(
            f"{context}: action {index} must be a JSON object with key \"op\", "
            f"got {type(action).__name__}."
        )
    out = dict(action)
    if "op" not in out:
        for alias in ("type", "action", "name", "operation"):
            if alias in out and isinstance(out[alias], str):
                out["op"] = out.pop(alias)
                break
    if "op" not in out and {"x", "y", "z"}.issubset(out.keys()):
        out["op"] = "arm_translate"
    if "op" not in out and ("xyz" in out or "rpy_deg" in out):
        out["op"] = "arm_goto_pose"
    params = out.pop("parameters", None)
    if isinstance(params, dict):
        for key, val in params.items():
            out.setdefault(key, val)
    if "op" not in out:
        raise ValueError(
            f'{context}: action {index} must include top-level "op" '
            f'(not "type" or nested "parameters" only). Keys seen: {sorted(out)}.'
        )
    out["op"] = _canonicalize_op_name(str(out["op"]), out)
    return out


def _validate_actions_list(actions: Any, *, context: str) -> list[dict[str, Any]]:
    """Ensure executor-compatible ``{"op": ...}`` actions; return normalized list."""
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{context}: actions must be a non-empty list.")
    if len(actions) > _MAX_ACTIONS:
        raise ValueError(f"{context}: at most {_MAX_ACTIONS} actions.")
    normalized: list[dict[str, Any]] = []
    for i, raw in enumerate(actions):
        a = _normalize_action_dict(raw, context=context, index=i)
        op = a["op"]
        if op not in _ALLOWED_ACTION_OPS:
            raise ValueError(
                f"{context}: unknown op {op!r} in action {i}. "
                f"Use only: {', '.join(sorted(_ALLOWED_ACTION_OPS))}."
            )
        normalized.append(a)
    return normalized


def _validate_labels_schema(data: Any) -> list[str]:
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError('Label JSON must be {"labels": ["...", ...]}.')
    raw = data["labels"]
    if not isinstance(raw, list) or not raw:
        raise ValueError('"labels" must be a non-empty list of strings.')
    out = [str(x).strip() for x in raw if str(x).strip()]
    if not out:
        raise ValueError('"labels" must contain at least one non-empty string.')
    if len(out) > 12:
        raise ValueError("At most 12 labels allowed.")
    return out


def _validate_plan_schema(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or "subtasks" not in plan:
        raise ValueError("Plan JSON must be an object with key 'subtasks'.")
    subs = plan["subtasks"]
    if not isinstance(subs, list) or not subs:
        raise ValueError("'subtasks' must be a non-empty list.")
    if len(subs) > _MAX_SUBTASKS:
        raise ValueError(
            f"Too many subtasks ({len(subs)} > {_MAX_SUBTASKS}). "
            "Combine related motions into fewer subtasks with multiple actions each."
        )
    for si, s in enumerate(subs):
        if not isinstance(s, dict):
            raise ValueError("Each subtask must be an object.")
        if "description" not in s or "actions" not in s:
            raise ValueError("Each subtask needs 'description' and 'actions'.")
        acts = s["actions"]
        if not isinstance(acts, list) or not acts:
            raise ValueError("Each subtask needs a non-empty 'actions' list.")
        if len(acts) > _MAX_ACTIONS:
            raise ValueError(f"Each subtask allows at most {_MAX_ACTIONS} actions.")
        s["actions"] = _validate_actions_list(acts, context=f"subtasks[{si}]")
    return plan


class ReasoningClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        run_tracker: RunTracker | None = None,
    ) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to repo-root .env).")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._run_tracker = run_tracker
        self._client = OpenAI(timeout=60.0)

    def infer_tracked_labels(self, task: str, rgb: np.ndarray | None = None) -> list[str]:
        """Derive OWL-ViT / Perception label strings from the natural-language task (before detection)."""
        user_text = f"TASK:\n{task}\n"
        if rgb is not None:
            content: list[dict[str, Any]] | str = _user_multimodal_rgb_depth(user_text, rgb)
        else:
            content = user_text
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LABELS_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        if self._run_tracker is not None:
            self._run_tracker.record_llm_response("infer_tracked_labels", self.model, resp)
        raw = resp.choices[0].message.content or ""
        data = _parse_llm_json(raw, context="infer_tracked_labels")
        labels = _validate_labels_schema(data)
        logger.info("Inferred labels: %s", labels)
        return labels

    def generate_plan(
        self,
        task: str,
        scene_description: str,
        rgb: np.ndarray,
        *,
        depth_viz_rgb: np.ndarray | None = None,
        tracked_labels: Sequence[str] | None = None,
        robot_context: str | None = None,
        workspace_bounds: str | None = None,
        scene_hints: str | None = None,
    ) -> dict[str, Any]:
        user_text = f"TASK:\n{task}\n\n"
        if robot_context:
            user_text += f"{robot_context}\n\n"
        user_text += f"PERCEPTION:\n{scene_description}\n"
        if scene_hints:
            user_text += f"\n{scene_hints}\n"
        if workspace_bounds:
            user_text += f"\n{workspace_bounds}\n"
        if tracked_labels:
            user_text += f"\nVISION_LABELS: {', '.join(str(x) for x in tracked_labels)}\n"
        content = _user_multimodal_rgb_depth(user_text, rgb, depth_viz_rgb)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        if self._run_tracker is not None:
            self._run_tracker.record_llm_response("generate_plan", self.model, resp)
        choice = resp.choices[0]
        message = choice.message
        raw = message.content or ""
        if not raw.strip():
            refusal = getattr(message, "refusal", None)
            fr = getattr(choice, "finish_reason", None)
            raise ValueError(
                "generate_plan: empty model content "
                f"(finish_reason={fr!r}, refusal={refusal!r}). "
                "Try another model or check moderation / API errors."
            )
        logger.debug("generate_plan raw response: %s", raw[:2000])
        data = _parse_llm_json(raw, context="generate_plan")
        return _validate_plan_schema(data)

    def verify_execution(
        self,
        *,
        goal: str,
        subtask_description: str,
        scene_after: str,
        rgb_after: np.ndarray,
        verification_mode: str = "grasp",
        rgb_before: np.ndarray | None = None,
        depth_viz_after: np.ndarray | None = None,
        depth_viz_before: np.ndarray | None = None,
    ) -> dict[str, Any]:
        parts = [
            f"GOAL:\n{goal}\n",
            f"SUBTASK (just attempted):\n{subtask_description}\n",
            f"SUBTASK_MODE: {verification_mode}\n",
            f"PERCEPTION_AFTER:\n{scene_after}\n",
        ]
        content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(parts)}]
        if rgb_before is not None:
            content.extend(
                _user_multimodal_rgb_depth(
                    "BEFORE subtask (RGB + depth):",
                    rgb_before,
                    depth_viz_before,
                )[1:]
            )
        content.extend(
            _user_multimodal_rgb_depth(
                "AFTER subtask (RGB + depth):",
                rgb_after,
                depth_viz_after,
            )[1:]
        )

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        if self._run_tracker is not None:
            self._run_tracker.record_llm_response("verify_execution", self.model, resp)
        raw = resp.choices[0].message.content or ""
        data = _parse_llm_json(raw, context="verify_execution")
        if "status" not in data:
            raise ValueError("Verification JSON must include 'status'.")
        if data["status"] not in ("OK", "FAILED"):
            raise ValueError("Verification status must be OK or FAILED.")
        reason = data.get("failure_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("failure_reason must be a string or null.")
        corr = data.get("correction")
        if corr is not None:
            if not isinstance(corr, dict) or "actions" not in corr:
                raise ValueError("correction must be null or an object with 'actions'.")
            acts = corr["actions"]
            if not isinstance(acts, list) or len(acts) > _MAX_ACTIONS:
                raise ValueError(f"correction.actions must be a list of at most {_MAX_ACTIONS} items.")
            try:
                corr["actions"] = _validate_actions_list(acts, context="verify_execution.correction")
            except ValueError as exc:
                logger.warning("Dropping invalid verify correction (will not run on robot): %s", exc)
                data["correction"] = None
        return data

    def replan_after_failure(
        self,
        *,
        task: str,
        failed_subtask_description: str,
        failed_subtask: dict[str, Any],
        verification: dict[str, Any],
        scene_description: str,
        rgb: np.ndarray,
        robot_context: str | None = None,
        workspace_bounds: str | None = None,
        scene_hints: str | None = None,
        depth_viz_rgb: np.ndarray | None = None,
        tracked_labels: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Recovery plan from post-failure perception (re-approach, re-grasp, etc.)."""
        import json as _json

        user_text = (
            f"TASK:\n{task}\n\n"
            f"FAILED_SUBTASK:\n{failed_subtask_description}\n\n"
            f"FAILED_SUBTASK_JSON:\n{_json.dumps(failed_subtask, indent=2)}\n\n"
            f"VERIFICATION:\n{_json.dumps(verification, indent=2)}\n\n"
        )
        if robot_context:
            user_text += f"{robot_context}\n\n"
        user_text += f"PERCEPTION_AFTER:\n{scene_description}\n"
        if scene_hints:
            user_text += f"\n{scene_hints}\n\n"
        if workspace_bounds:
            user_text += f"{workspace_bounds}\n\n"
        if tracked_labels:
            user_text += f"VISION_LABELS: {', '.join(tracked_labels)}\n"
        content = _user_multimodal_rgb_depth(user_text, rgb, depth_viz_rgb)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REPLAN_AFTER_FAILURE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        if self._run_tracker is not None:
            self._run_tracker.record_llm_response("replan_after_failure", self.model, resp)
        raw = resp.choices[0].message.content or ""
        data = _parse_llm_json(raw, context="replan_after_failure")
        plan = _validate_plan_schema(data)
        if len(plan["subtasks"]) > 2:
            raise ValueError("Recovery replan must have at most 2 subtasks.")
        return plan

    def replan_out_of_workspace(
        self,
        *,
        task: str,
        scene_description: str,
        rgb: np.ndarray,
        rejected_xyz: tuple[float, float, float],
        bounds: dict[str, float],
        robot_context: str | None = None,
        depth_viz_rgb: np.ndarray | None = None,
        scene_hints: str | None = None,
    ) -> dict[str, Any]:
        user_text = f"TASK:\n{task}\n\n"
        if robot_context:
            user_text += f"{robot_context}\n\n"
        user_text += f"PERCEPTION:\n{scene_description}\n"
        if scene_hints:
            user_text += f"\n{scene_hints}\n\n"
        user_text += (
            f"REJECTED_TARGET_XYZ (robot frame, meters): {list(rejected_xyz)}\n"
            f"ALLOWED_AXIS_ALIGNED_BOX: xmin={bounds['xmin']}, xmax={bounds['xmax']}, "
            f"ymin={bounds['ymin']}, ymax={bounds['ymax']}, zmin={bounds['zmin']}, zmax={bounds['zmax']}\n"
        )
        content = _user_multimodal_rgb_depth(user_text, rgb, depth_viz_rgb)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REPLAN_OOB_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        if self._run_tracker is not None:
            self._run_tracker.record_llm_response("replan_out_of_workspace", self.model, resp)
        raw = resp.choices[0].message.content or ""
        data = _parse_llm_json(raw, context="replan_out_of_workspace")
        return _validate_plan_schema(data)
