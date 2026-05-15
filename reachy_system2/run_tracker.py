"""Per-run artifacts under ``reachy_system2/runs/<timestamp>/`` (token usage, metadata)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _runs_root() -> Path:
    return Path(__file__).resolve().parent / "runs"


def _serialize_usage(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    out: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "completion_tokens_details",
        "prompt_tokens_details",
    ):
        val = getattr(usage, key, None)
        if val is not None:
            if hasattr(val, "model_dump"):
                out[key] = val.model_dump()
            else:
                out[key] = val
    return out


def _aggregate_usage(calls: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in calls:
        u = row.get("usage") or {}
        for k in totals:
            try:
                totals[k] += int(u.get(k) or 0)
            except (TypeError, ValueError):
                pass
    return totals


@dataclass
class RunTracker:
    """Create a directory for one closed-loop run and append LLM usage records."""

    run_id: str | None = None
    root: Path = field(default_factory=_runs_root)

    def __post_init__(self) -> None:
        rid = self.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = self.root / rid
        self.dir.mkdir(parents=True, exist_ok=True)
        self._calls: list[dict[str, Any]] = []
        self._llm_log_path = self.dir / "llm_calls.jsonl"
        logger.info("Run artifacts directory: %s", self.dir)

    @property
    def path(self) -> Path:
        return self.dir

    def write_run_meta(self, meta: dict[str, Any]) -> None:
        payload = {
            **meta,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (self.dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_text_file(self, name: str, content: str) -> None:
        (self.dir / name).write_text(content, encoding="utf-8")

    def write_json(self, name: str, data: Any) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_llm_response(self, call: str, model: str, response: Any) -> None:
        usage = _serialize_usage(getattr(response, "usage", None))
        row: dict[str, Any] = {
            "call": call,
            "model": model,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "usage": usage,
        }
        self._calls.append(row)
        with self._llm_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def finalize(self, *, status: str, error: str | None = None) -> None:
        totals = _aggregate_usage(self._calls)
        summary = {
            "status": status,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm_call_count": len(self._calls),
            "token_usage_totals": totals,
            "per_call": self._calls,
        }
        if error:
            summary["error"] = error
        (self.dir / "token_usage.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        meta_path = self.dir / "run.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        else:
            meta = {}
        meta["finished_at_utc"] = summary["finished_at_utc"]
        meta["status"] = status
        meta["token_usage_totals"] = totals
        meta["llm_call_count"] = len(self._calls)
        if error:
            meta["error"] = error
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def log_runs_enabled() -> bool:
    return os.environ.get("SYSTEM2_LOG_RUNS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
