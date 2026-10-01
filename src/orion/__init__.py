"""AeroNetra-Orion — event-driven predictive perception for UAVs.

The package couples three ideas that are usually studied separately:

1. **Neuromorphic event streams** (microsecond latency, high dynamic range,
   no motion blur) simulated from or loaded from real event sensors.
2. **Event-guided fusion** that restores detail in motion-blurred frames using
   the asynchronous event signal.
3. A **predictive world model** that forecasts short-horizon object motion and
   converts it into a *time-to-collision* map for anticipatory UAV avoidance.

Public API is exposed lazily so that ``import orion`` stays lightweight.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "OrionConfig", "load_config", "OrionPipeline"]


def __getattr__(name: str):
    if name in ("OrionConfig", "load_config", "config_from_dict", "to_dict"):
        from . import config as _config

        return getattr(_config, name)
    if name in ("OrionPipeline", "run_demo"):
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module 'orion' has no attribute {name!r}")
