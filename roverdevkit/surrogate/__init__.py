"""Mission-level surrogate as an acceleration and uncertainty layer.

**Reframed scope (project_log.md 2026-04-26).** After the W7.7 traverse-loop
lift-out the corrected mission evaluator runs at ~40 ms / mission, so this
package is no longer the project's "fast path." It is an *optional* layer
on top of the corrected evaluator that exists for:

1. NSGA-II inner-loop fitness function (≈30k evals × 4 scenarios is
   surrogate territory; the corrected evaluator validates the final
   Pareto front).
2. Bulk sensitivity / Sobol / 1M-point grids where 40 ms × N becomes
   uncomfortable.
3. Calibrated 90 % prediction intervals via quantile XGBoost (W8 step-4).
4. Probabilistic feasibility for NSGA-II constraint handling
   (classifier AUC, not deterministic boolean).
5. Phase-5 benchmark baseline.
6. Deployment portability (a pickled XGBoost is much smaller than the
   full evaluator stack).

The wheel-level multi-fidelity correction lives elsewhere — see
:class:`roverdevkit.terramechanics.correction_model.WheelLevelCorrection`
— and is composed into the analytical traverse loop directly, not via
this package.

Modules:

- :mod:`.sampling` — stratified Latin-Hypercube sampler over the 12-D
  design space × 4 scenario families with jittered scenario/soil
  parameters.
- :mod:`.dataset` — parallel dataset builder + Parquet I/O. Consumes
  ``LHSSample`` from :mod:`.sampling` and produces a flat-schema
  DataFrame of evaluator outputs plus aggregate traverse-log
  statistics.
- :mod:`.features` — feature engineering (dimensionless groups,
  physics-informed transforms).
- :mod:`.baselines` — Ridge / RF / XGBoost per target + joint MLP +
  feasibility classifier. The default-hyperparameter pipeline used by
  W6 / W8 step-2.
- :mod:`.tuning` — Optuna TPE on XGBoost (W8 step-3).
- :mod:`.metrics` — R²/RMSE/MAPE, AUC/F1, per-scenario-family
  breakdowns, the canonical ``benchmark_score`` API.

Target accuracy (project_plan.md §7 Layer 1, surrogate vs corrected
evaluator):
    R² > 0.95 for range_km and energy_margin_raw_pct; R² > 0.85 for
    slope_capability_deg and total_mass_kg; AUC > 0.90 for
    stalled feasibility.
"""

from roverdevkit.surrogate.baselines import (
    ACCEPTANCE_GATES,
    CLASSIFIER_ALGORITHMS,
    JOINT_MLP_NAME,
    REGRESSION_ALGORITHMS,
    FittedBaselines,
    acceptance_gate,
    evaluate_baselines,
    fit_baselines,
    predict_for_registry_rovers,
)
from roverdevkit.surrogate.dataset import (
    DatasetMetadata,
    build_and_write,
    build_dataset,
    read_parquet,
    read_parquet_metadata,
    write_parquet,
)
from roverdevkit.surrogate.sampling import (
    FAMILIES,
    LHSSample,
    ScenarioFamily,
    generate_samples,
)

__all__ = [
    "ACCEPTANCE_GATES",
    "CLASSIFIER_ALGORITHMS",
    "FAMILIES",
    "DatasetMetadata",
    "FittedBaselines",
    "JOINT_MLP_NAME",
    "LHSSample",
    "REGRESSION_ALGORITHMS",
    "ScenarioFamily",
    "acceptance_gate",
    "build_and_write",
    "build_dataset",
    "evaluate_baselines",
    "fit_baselines",
    "generate_samples",
    "predict_for_registry_rovers",
    "read_parquet",
    "read_parquet_metadata",
    "write_parquet",
]
