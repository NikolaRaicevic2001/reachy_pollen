"""OpenAI vision client: planning, verification, out-of-workspace replanning."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Sequence

import certifi
import httpx
import numpy as np
from openai import OpenAI

from reachy_system2.prompts import (
    PLAN_SYSTEM_PROMPT,
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


def _user_multimodal_text_image(text: str, rgb: np.ndarray) -> list[dict[str, Any]]:
    b64 = _encode_image_jpeg_b64(rgb)
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]


def _validate_actions_list(actions: Any, *, context: str) -> None:
    """Ensure executor-compatible ``{"op": ...}`` actions."""
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{context}: actions must be a non-empty list.")
    if len(actions) > _MAX_ACTIONS:
        raise ValueError(f"{context}: at most {_MAX_ACTIONS} actions.")
    for i, a in enumerate(actions):
        if not isinstance(a, dict) or "op" not in a:
            raise ValueError(f'{context}: action {i} must be an object with an "op" field.')
        op = a["op"]
        if op not in _ALLOWED_ACTION_OPS:
            raise ValueError(
                f"{context}: unknown op {op!r} in action {i}. "
                f"Use only: {', '.join(sorted(_ALLOWED_ACTION_OPS))}."
            )


def _validate_plan_schema(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or "subtasks" not in plan:
        raise ValueError("Plan JSON must be an object with key 'subtasks'.")
    subs = plan["subtasks"]
    if not isinstance(subs, list) or not subs:
        raise ValueError("'subtasks' must be a non-empty list.")
    if len(subs) > _MAX_SUBTASKS:
        raise ValueError(
            f"Too many subtasks ({len(subs)} > {_MAX_SUBTASKS}). Merge phases: prefer ~4–6 subtasks "
            "with 2–5 actions each instead of many single-action subtasks."
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
        _validate_actions_list(acts, context=f"subtasks[{si}]")
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
        _base = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        # certifi CA bundle: avoids SSL: CERTIFICATE_VERIFY_FAILED on some Windows/Python installs
        # where the default trust store is incomplete (SSL_CERT_FILE alone may not reach httpx).
        _http = httpx.Client(verify=certifi.where(), timeout=60.0)
        self._client = OpenAI(http_client=_http, base_url=_base) if _base else OpenAI(http_client=_http)

    def generate_plan(
        self,
        task: str,
        scene_description: str,
        rgb: np.ndarray,
        *,
        tracked_labels: Sequence[str] | None = None,
        robot_context: str | None = None,
    ) -> dict[str, Any]:
        extras: list[str] = []
        if tracked_labels:
            extras.append(
                "TRACKED_LABELS (vision system): " + ", ".join(str(x) for x in tracked_labels)
            )
        has_object_lines = any(
            ln.strip().startswith("- ") and ": x=" in ln for ln in scene_description.splitlines()
        )
        if "(none above threshold)" in scene_description or not has_object_lines:
            extras.append(
                "NOTE: No usable 3D object lines are in PERCEPTION. Ground yourself in the RGB image. "
                "You MUST output ONLY valid JSON as specified — no questions, no prose outside JSON. "
                "Propose conservative Cartesian targets consistent with the workspace."
            )
        extra_block = ("\n".join(extras) + "\n") if extras else ""
        user_text = f"TASK:\n{task}\n\nPERCEPTION:\n{scene_description}\n"
        if robot_context:
            user_text += f"\n{robot_context}\n"
        if extra_block:
            user_text += "\n" + extra_block
        user_text += "\nHint: few subtasks, 2–5 ops each.\n"
        content = _user_multimodal_text_image(user_text, rgb)
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
        rgb_before: np.ndarray | None = None,
    ) -> dict[str, Any]:
        parts = [
            f"GOAL:\n{goal}\n",
            f"SUBTASK (just attempted):\n{subtask_description}\n",
            f"PERCEPTION_AFTER:\n{scene_after}\n",
        ]
        content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(parts)}]
        if rgb_before is not None:
            content.append({"type": "text", "text": "IMAGE_BEFORE (before subtask):"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_image_jpeg_b64(rgb_before)}"
                    },
                }
            )
        content.append({"type": "text", "text": "IMAGE_AFTER (after subtask):"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_jpeg_b64(rgb_after)}"},
            },
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
        corr = data.get("correction")
        if corr is not None:
            if not isinstance(corr, dict) or "actions" not in corr:
                raise ValueError("correction must be null or an object with 'actions'.")
            acts = corr["actions"]
            if not isinstance(acts, list) or len(acts) > _MAX_ACTIONS:
                raise ValueError(f"correction.actions must be a list of at most {_MAX_ACTIONS} items.")
            _validate_actions_list(acts, context="verify_execution.correction")
        return data

    def replan_out_of_workspace(
        self,
        *,
        task: str,
        scene_description: str,
        rgb: np.ndarray,
        rejected_xyz: tuple[float, float, float],
        bounds: dict[str, float],
        robot_context: str | None = None,
    ) -> dict[str, Any]:
        user_text = (
            f"GOAL:\n{task}\n\nPERCEPTION:\n{scene_description}\n\n"
            f"REJECTED_TARGET_XYZ (robot frame, meters): {list(rejected_xyz)}\n"
            f"ALLOWED_AXIS_ALIGNED_BOX: xmin={bounds['xmin']}, xmax={bounds['xmax']}, "
            f"ymin={bounds['ymin']}, ymax={bounds['ymax']}, zmin={bounds['zmin']}, zmax={bounds['zmax']}\n"
        )
        if robot_context:
            user_text += f"\n{robot_context}\n"
        user_text += "\nHint: few subtasks, bundle ops.\n"
        content = _user_multimodal_text_image(user_text, rgb)
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
