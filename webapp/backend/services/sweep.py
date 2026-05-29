"""Backend dispatcher for parametric sweeps.

Pure-Python core lives in :mod:`roverdevkit.tradespace.sweeps` (axis
definitions, grid expansion, backend picking). This module is the
glue that loads the artifacts (correction model, quantile bundles,
soil parameters), runs the chosen backend, and returns a
:class:`~roverdevkit.tradespace.sweeps.SweepResult`.

Two backends, two performance profiles
--------------------------------------
- **Evaluator**: corrected analytical pipeline + wheel-level SCM
  correction. ~40 ms / cell after the BW/SCM bake-off lift-out. Ground truth.
  Used for ≤ ``EVALUATOR_AUTO_THRESHOLD`` cells in auto mode.
- **Surrogate**: τ=0.5 head from the quantile bundles. Vectorised --
  one batch ``predict`` over the whole grid. ~5 ms total for any
  reasonable resolution. Used for > ``EVALUATOR_AUTO_THRESHOLD``
  cells in auto mode.

Both backends emit values for the same primary regression targets so
the response shape is identical.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.correction_model import WheelLevelCorrection
from roverdevkit.tradespace.sweeps import (
    SweepResult,
    SweepSpec,
    expand_grid,
    pick_backend,
)
from webapp.backend.services.predict import build_feature_row


def run_sweep(
    spec: SweepSpec,
    base_design: DesignVector,
    scenario: MissionScenario,
    soil: SoilParameters,
    *,
    correction: WheelLevelCorrection | None,
    bundles: dict[str, QuantileHeads],
) -> SweepResult:
    """Resolve the backend, execute the sweep, return a packed result.

    Parameters
    ----------
    spec
        Validated sweep specification (axes + target + backend mode).
    base_design
        The "rest of the design" -- every dimension not on an axis is
        held at this value across the whole grid.
    scenario, soil
        The mission scenario (already resolved to one of the canonical
        four) and its nominal soil parameters. Both are constant
        across the grid -- a sweep varies design, not scenario.
    correction
        Wheel-level SCM correction artifact for the evaluator path.
        Pass ``None`` to fall back to BW-only; the route is
        responsible for surfacing the degraded mode in the response.
    bundles
        Quantile XGBoost bundles for the surrogate path. Required
        even when the auto-picker chooses the evaluator -- the route
        loads them once per process and passes them through unchanged.
    """
    backend = pick_backend(spec)
    designs = expand_grid(spec, base_design)

    t0 = time.perf_counter()
    if backend == "evaluator":
        z_flat = _run_evaluator(spec, designs, scenario, correction=correction)
    elif backend == "surrogate":
        z_flat = _run_surrogate(spec, designs, scenario, soil, bundles=bundles)
    else:  # pragma: no cover -- pick_backend guards this
        raise AssertionError(f"unreachable backend {backend!r}")
    elapsed_s = time.perf_counter() - t0

    x_values = spec.x_axis.values()
    y_values = spec.y_axis.values() if spec.y_axis is not None else None
    if y_values is None:
        z_values = z_flat
    else:
        # expand_grid emits row-major (y outer, x inner); reshape so
        # the first axis is y to match Plotly heatmap orientation.
        z_values = z_flat.reshape(spec.y_axis.n_points, spec.x_axis.n_points)

    return SweepResult(
        spec=spec,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
        backend_used=backend,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Evaluator path
# ---------------------------------------------------------------------------


def _run_evaluator(
    spec: SweepSpec,
    designs: list[DesignVector],
    scenario: MissionScenario,
    *,
    correction: WheelLevelCorrection | None,
) -> np.ndarray:
    """Per-cell call to :func:`roverdevkit.mission.evaluator.evaluate`.

    Returns a 1-D array of length ``len(designs)`` (already in
    row-major order; the caller reshapes for 2-D sweeps).
    """
    use_corr = correction is not None
    out = np.empty(len(designs), dtype=float)
    for i, d in enumerate(designs):
        metrics = evaluator_evaluate(
            d,
            scenario,
            use_scm_correction=use_corr,
            correction=correction,
        )
        out[i] = float(getattr(metrics, spec.target))
    return out


# ---------------------------------------------------------------------------
# Surrogate path
# ---------------------------------------------------------------------------


def _run_surrogate(
    spec: SweepSpec,
    designs: list[DesignVector],
    scenario: MissionScenario,
    soil: SoilParameters,
    *,
    bundles: dict[str, QuantileHeads],
) -> np.ndarray:
    """Vectorised batch predict on the τ=0.5 head of the chosen target.

    Builds one feature DataFrame for the entire grid and runs a
    single ``QuantileHeads.predict`` call -- XGBoost's batched
    prediction is dramatically faster than per-row calls and dwarfs
    the per-row feature-construction time at any reasonable grid size.
    """
    if spec.target not in bundles:
        raise KeyError(
            f"quantile bundle missing target {spec.target!r}; "
            f"available: {sorted(bundles.keys())}."
        )
    feature_rows = [
        build_feature_row(d, scenario, soil) for d in designs
    ]
    X = pd.concat(feature_rows, ignore_index=True)
    preds = bundles[spec.target].predict(X, repair_crossings=True)
    # The "0.50" key is added by QuantileHeads.predict for whichever
    # quantile equals 0.5 in the configured triple. Default triple is
    # (0.05, 0.5, 0.95), so this lookup matches the quantile-calibration contract.
    if "q50" in preds:
        return np.asarray(preds["q50"], dtype=float)
    # Fallback: pick the entry whose label is closest to 0.5. Defensive
    # against future bundle versions that stash predictions under a
    # numeric key.
    closest = min(preds.keys(), key=lambda k: abs(_quantile_from_key(k) - 0.5))
    return np.asarray(preds[closest], dtype=float)


def _quantile_from_key(key: str) -> float:
    """Parse ``"q05"`` -> 0.05, ``"q50"`` -> 0.5, etc.; ``"0.5"`` also works."""
    if key.startswith("q"):
        return int(key[1:]) / 100.0
    return float(key)
