"""System 2 closed-loop: perception (Pollen Vision), LLM reasoning, guarded execution."""

from reachy_system2.perception import (
    PerceptionSnapshot,
    System2Perception,
    missing_tracked_labels,
)

__version__ = "0.1.0"
__all__ = [
    "PerceptionSnapshot",
    "System2Perception",
    "missing_tracked_labels",
    "__version__",
]
