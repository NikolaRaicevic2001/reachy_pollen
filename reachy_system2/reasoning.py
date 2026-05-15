"""OpenAI vision client: planning, verification, out-of-workspace replanning."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any

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


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


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


def _validate_plan_schema(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or "subtasks" not in plan:
        raise ValueError("Plan JSON must be an object with key 'subtasks'.")
    subs = plan["subtasks"]
    if not isinstance(subs, list) or not subs:
        raise ValueError("'subtasks' must be a non-empty list.")
    for s in subs:
        if not isinstance(s, dict):
            raise ValueError("Each subtask must be an object.")
        if "description" not in s or "actions" not in s:
            raise ValueError("Each subtask needs 'description' and 'actions'.")
        acts = s["actions"]
        if not isinstance(acts, list) or not acts:
            raise ValueError("Each subtask needs a non-empty 'actions' list.")
        if len(acts) > _MAX_ACTIONS:
            raise ValueError(f"Each subtask allows at most {_MAX_ACTIONS} actions.")
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

    def generate_plan(self, task: str, scene_description: str, rgb: np.ndarray) -> dict[str, Any]:
        user_text = f"TASK:\n{task}\n\nPERCEPTION:\n{scene_description}\n"
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
        raw = resp.choices[0].message.content or ""
        logger.debug("generate_plan raw response: %s", raw[:2000])
        data = json.loads(_strip_json_fence(raw))
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
        data = json.loads(_strip_json_fence(raw))
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
        return data

    def replan_out_of_workspace(
        self,
        *,
        task: str,
        scene_description: str,
        rgb: np.ndarray,
        rejected_xyz: tuple[float, float, float],
        bounds: dict[str, float],
    ) -> dict[str, Any]:
        user_text = (
            f"GOAL:\n{task}\n\nPERCEPTION:\n{scene_description}\n\n"
            f"REJECTED_TARGET_XYZ (robot frame, meters): {list(rejected_xyz)}\n"
            f"ALLOWED_AXIS_ALIGNED_BOX: xmin={bounds['xmin']}, xmax={bounds['xmax']}, "
            f"ymin={bounds['ymin']}, ymax={bounds['ymax']}, zmin={bounds['zmin']}, zmax={bounds['zmax']}\n"
        )
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
        data = json.loads(_strip_json_fence(raw))
        return _validate_plan_schema(data)
