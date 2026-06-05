"""RoverDevKit — ML-accelerated co-design of lunar micro-rover mobility and power.

Top-level package. Most public API lives in submodules:

- :mod:`roverdevkit.schema` — shared dataclasses for design vectors, scenarios,
  and mission metrics.
- :mod:`roverdevkit.terramechanics` — Bekker-Wong analytical terramechanics.
- :mod:`roverdevkit.power` — solar, battery, and thermal survival sub-models.
- :mod:`roverdevkit.mass` — parametric mass-estimating relationships.
- :mod:`roverdevkit.mission` — top-level mission evaluator, scenarios,
  time-stepped traverse simulator.
- :mod:`roverdevkit.surrogate` — training, models, feature engineering, UQ.
- :mod:`roverdevkit.tradespace` — sweeps, NSGA-II optimization, SHAP rules.
- :mod:`roverdevkit.validation` — rover rediscovery, experimental comparison,
  error budget.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
