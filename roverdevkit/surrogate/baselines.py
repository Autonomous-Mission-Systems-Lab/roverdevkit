"""Week-6 baseline surrogate models (project_plan.md §6 step-4).

This module trains and evaluates the four canonical baseline families
the Week-6 acceptance gate is reported against:

- **Ridge** — linear baseline; serves as the floor any non-linear model
  must beat.
- **Random Forest** — non-linear, no extrapolation, robust to scale and
  categorical encoding choice.
- **XGBoost** — primary baseline; consumes pandas ``category`` columns
  natively via ``enable_categorical=True``.
- **Joint MLP** (``sklearn.neural_network.MLPRegressor`` with shared
  hidden layers and one output neuron per target) — the only baseline
  that can share representations across the four regression targets.
  Reported as a single multi-output model rather than four per-target
  fits.

Per-target vs. joint
--------------------
For Ridge, RF, and XGBoost we fit **one model per target** rather than
a joint multi-output model. The four primary targets (``range_km``,
``energy_margin_raw_pct``, ``slope_capability_deg``, ``total_mass_kg``)
have very different physics and respond best to different
hyperparameters; a joint sklearn ``MultiOutputRegressor`` would
under-fit the harder ones and over-fit the easier ones with shared
hyperparams. The MLP is the exception precisely because shared hidden
layers are its main reason to exist; we keep it joint and let the
results speak.

Feasibility classifier
----------------------
A single-target binary classifier on ``stalled`` (schema v6 —
see ``data/analytical/SCHEMA.md``). Positive class = stalled =
infeasible. Trained on **all** rows (both feasible and infeasible)
because that is the population the deployed surrogate sees at
NSGA-II constraint evaluation time. Reported as AUC and F1 (both
overall and per scenario family).

Two-stage convention
--------------------
At evaluation time, the regressors are trained on the **feasible**
subset (``stalled == False``) so they don't waste capacity
modelling the ``range_km ~ 0`` failure mode; the feasibility
classifier is trained on **all** rows so the deployed surrogate can
gate predictions before the regressor ever runs.

Reproducibility
---------------
All randomised model components (RF, XGBoost subsampling, MLP weight
init, MLP train/val split) are seeded from the ``random_state``
parameter on :func:`fit_baselines`.

Hyperparameter tuning is intentionally **deferred to Week 7** so the
Week-6 numbers report sensible-default performance and the Week-7
Optuna lift is cleanly attributable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    PRIMARY_REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
    SCENARIO_NUMERIC_COLUMNS,
    build_feature_matrix,
    valid_rows,
)

REGRESSION_ALGORITHMS: tuple[str, ...] = ("ridge", "random_forest", "xgboost")
"""Per-target regression baselines (one fit per (algo, target))."""

CLASSIFIER_ALGORITHMS: tuple[str, ...] = ("logreg", "xgboost")
"""Feasibility-classifier baselines (one fit per algo)."""

JOINT_MLP_NAME: str = "mlp_joint"
"""Sentinel key used for the joint multi-output MLP regressor.

It does **not** appear in :data:`REGRESSION_ALGORITHMS` because it is
trained once (across all primary targets) rather than once per target;
keeping it under a separate key makes the per-(algo, target) loop
unambiguous."""

LAYER1_PRIMARY_TARGETS: tuple[str, ...] = (
    "total_mass_kg",
    "slope_capability_deg",
    "stalled",
)
"""Primary acceptance set for the registry-rover Layer-1 sanity check
(``predict_for_registry_rovers``). These three metrics depend on the
rover's *design* vector — chassis + wheels for mass, soil + wheel
geometry for slope capability, motor sizing × terramechanics for
feasibility — and the v3 widened LHS bounds put every flown / design-
target rover in the registry inside the surrogate's training support
on these dimensions. They are the metrics on which the Layer-1 sanity
check is treated as a real accuracy gate.

The two excluded targets are :data:`LAYER1_DIAGNOSTIC_TARGETS`."""

LAYER1_DIAGNOSTIC_TARGETS: tuple[str, ...] = (
    "range_km",
    "energy_margin_raw_pct",
)
"""Targets emitted by the Layer-1 sanity check for diagnostic purposes
only and explicitly excluded from the primary acceptance set.

Both are *scenario*-OOD for the registry rovers: their published
mission distances (Pragyan ≈ 100 m, Yutu-2 ≈ 25 m / lunar day,
MoonRanger ≈ 1 km / Earth-day, Rashid-1 ≈ 1 km) are 100-1000x smaller
than the LHS family traverse-distance budgets (20-80 km, intentionally
non-binding so ``range_km`` stays a continuous signal during training).
The surrogate's predictions live in the family-budget regime; the
Layer-1 truth values are the much smaller registry-scenario evaluator
outputs, so the relative error is dominated by an absolute-scale
mismatch with no bearing on physical model accuracy.

See ``data/analytical/SCHEMA.md`` v3 entry and ``project_log.md``
2026-04-25 LHS v3 entry for the full diagnosis."""

# Numeric columns that must be scaled for Ridge / MLP. Tree models
# (RF, XGB) don't care, but the same preprocessor is used for them so
# the column-trim step is uniform; the cost of scaling is negligible.
_NUMERIC_FOR_PREPROC: list[str] = [
    "design_wheel_radius_m",
    "design_wheel_width_m",
    "design_grouser_height_m",
    "design_grouser_count",
    "design_n_wheels",
    "design_chassis_mass_kg",
    "design_wheelbase_m",
    "design_solar_area_m2",
    "design_battery_capacity_wh",
    "design_avionics_power_w",
    "design_peak_wheel_torque_nm",
    *SCENARIO_NUMERIC_COLUMNS,
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FittedBaselines:
    """Bundle of fitted models produced by :func:`fit_baselines`.

    Attributes
    ----------
    regressors
        ``{(algo, target): fitted_estimator}`` for the per-target
        regression baselines (Ridge, RF, XGBoost).
    joint_mlp
        Single fitted multi-output MLP across :data:`mlp_targets`, or
        ``None`` if MLP training was skipped.
    mlp_targets
        Ordered tuple of regression targets the joint MLP was trained
        against. Empty if ``joint_mlp is None``.
    classifiers
        ``{algo: fitted_estimator}`` for the feasibility classifiers
        on :data:`FEASIBILITY_COLUMN`.
    fit_seconds
        ``{(algo, target | "<classifier>"): seconds}`` wall-clock per
        fit. Useful for the Week-6 writeup; not used by downstream
        evaluation.
    """

    regressors: dict[tuple[str, str], Any]
    joint_mlp: Any | None
    mlp_targets: tuple[str, ...]
    classifiers: dict[str, Any]
    fit_seconds: dict[tuple[str, str], float] = field(default_factory=dict)
    training_categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Per-categorical-column tuple of levels seen in training.

    Used by :func:`predict_for_registry_rovers` to conform the
    registry-rover input row to the training codebook so XGBoost's
    strict ``enable_categorical=True`` recode does not raise on an
    unseen simulant or terrain class. Unseen levels become NaN, which
    XGBoost treats as a missing category."""


# ---------------------------------------------------------------------------
# Preprocessor + estimator factories
# ---------------------------------------------------------------------------


def _make_preprocessor(*, scale_numerics: bool) -> ColumnTransformer:
    """ColumnTransformer used by every non-XGBoost baseline.

    Numeric columns are passed through (Ridge / MLP) or pre-scaled to
    zero-mean/unit-variance (RF doesn't strictly need it but the cost
    is negligible and the transformer stays uniform). Categoricals are
    one-hot encoded with ``handle_unknown='ignore'`` so registry-rover
    inference, which can in principle produce a category not seen in
    LHS training (e.g. an unfamiliar terrain class), does not crash.
    """
    numeric_step = StandardScaler() if scale_numerics else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, _NUMERIC_FOR_PREPROC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                SCENARIO_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_regressor(algo: str, *, random_state: int, n_jobs: int) -> Any:
    """Build a single per-target regression estimator.

    Returns a sklearn-compatible object that supports ``fit(X, y)`` /
    ``predict(X)`` directly on the full feature DataFrame (including
    pandas ``category`` columns).
    """
    if algo == "ridge":
        return Pipeline(
            [
                ("pre", _make_preprocessor(scale_numerics=True)),
                ("est", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
    if algo == "random_forest":
        return Pipeline(
            [
                ("pre", _make_preprocessor(scale_numerics=False)),
                (
                    "est",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=n_jobs,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if algo == "xgboost":
        # Native categorical handling: skip the OneHotEncoder/Pipeline
        # and let XGBoost split on category codes directly. This is
        # both faster and a stronger baseline than a one-hot Ridge-style
        # encoding for the four scenario_* categorical columns.
        return xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    raise ValueError(f"unknown regression algorithm {algo!r}; valid: {REGRESSION_ALGORITHMS}")


def _make_classifier(algo: str, *, random_state: int, n_jobs: int) -> Any:
    """Build a feasibility-classifier estimator."""
    if algo == "logreg":
        # ``n_jobs`` was deprecated on LogisticRegression in sklearn 1.8;
        # the parallelism flag now lives on the solver. We accept the
        # ``n_jobs`` arg here for API symmetry with the other estimators
        # but intentionally don't pass it through.
        del n_jobs
        return Pipeline(
            [
                ("pre", _make_preprocessor(scale_numerics=True)),
                (
                    "est",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if algo == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    raise ValueError(f"unknown classification algorithm {algo!r}; valid: {CLASSIFIER_ALGORITHMS}")


def _make_joint_mlp(*, random_state: int) -> TransformedTargetRegressor:
    """Multi-output MLP with one shared hidden trunk and N output heads.

    Wrapped in a :class:`TransformedTargetRegressor` so the targets are
    standardised before training (the four primary targets span ~3
    orders of magnitude — ``total_mass_kg`` ~30, ``range_km`` ~100s,
    ``energy_margin_raw_pct`` can be ±100s — and an unscaled MSE loss
    would be dominated by the largest target).
    """
    base = Pipeline(
        [
            ("pre", _make_preprocessor(scale_numerics=True)),
            (
                "est",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())


# ---------------------------------------------------------------------------
# Public API: training
# ---------------------------------------------------------------------------


def fit_baselines(
    df_train: pd.DataFrame,
    *,
    targets: tuple[str, ...] = tuple(PRIMARY_REGRESSION_TARGETS),
    regression_algorithms: tuple[str, ...] = REGRESSION_ALGORITHMS,
    classifier_algorithms: tuple[str, ...] = CLASSIFIER_ALGORITHMS,
    fit_mlp: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> FittedBaselines:
    """Fit the full per-target × per-algorithm baseline matrix.

    Parameters
    ----------
    df_train
        Training DataFrame. Must include :data:`INPUT_COLUMNS`,
        ``status``, the chosen ``targets``, and
        :data:`FEASIBILITY_COLUMN`. Rows with ``status != 'ok'`` are
        dropped via :func:`valid_rows`. The feasibility classifier
        sees both feasible and infeasible (post-``status``) rows; the
        regressors only see ``stalled == False`` (i.e. feasible) rows.
    targets
        Regression targets to fit. Defaults to the four Week-6
        primary targets.
    regression_algorithms, classifier_algorithms
        Subsets of :data:`REGRESSION_ALGORITHMS` /
        :data:`CLASSIFIER_ALGORITHMS`.
    fit_mlp
        If True (default) also fit the joint multi-output MLP across
        ``targets``. Falsy lets a fast smoke test skip the ~30 s MLP
        cost.
    random_state, n_jobs
        Plumbed through to every estimator.
    """
    import time

    df_clean = valid_rows(df_train)
    # Schema v6: feasibility is "not stalled" (positive class flipped).
    feas_series = ~df_clean[FEASIBILITY_COLUMN].astype(bool)
    if verbose:
        print(
            f"[fit_baselines] training rows: {len(df_clean)} "
            f"(after status='ok' filter); feasible rows: "
            f"{int(feas_series.sum())}",
            flush=True,
        )
    X_all = build_feature_matrix(df_clean)

    # Regressors: train on the feasible (non-stalled) subset
    feas_mask = feas_series.to_numpy()
    X_feas = X_all.loc[feas_mask].copy()

    regressors: dict[tuple[str, str], Any] = {}
    fit_seconds: dict[tuple[str, str], float] = {}
    for algo in regression_algorithms:
        for target in targets:
            y_feas = df_clean.loc[feas_mask, target].to_numpy()
            est = _make_regressor(algo, random_state=random_state, n_jobs=n_jobs)
            t0 = time.perf_counter()
            est.fit(X_feas, y_feas)
            dt = time.perf_counter() - t0
            regressors[(algo, target)] = est
            fit_seconds[(algo, target)] = dt
            if verbose:
                print(f"  fit {algo:<14s} {target:<24s} -> {dt:6.2f}s", flush=True)

    joint_mlp: TransformedTargetRegressor | None = None
    mlp_targets: tuple[str, ...] = ()
    if fit_mlp:
        Y_feas = df_clean.loc[feas_mask, list(targets)].to_numpy()
        mlp = _make_joint_mlp(random_state=random_state)
        t0 = time.perf_counter()
        mlp.fit(X_feas, Y_feas)
        dt = time.perf_counter() - t0
        joint_mlp = mlp
        mlp_targets = tuple(targets)
        fit_seconds[(JOINT_MLP_NAME, "joint")] = dt
        if verbose:
            print(f"  fit {JOINT_MLP_NAME:<14s} {'(joint)':<24s} -> {dt:6.2f}s", flush=True)

    # Classifiers: train on all clean rows (both feasible and infeasible)
    classifiers: dict[str, Any] = {}
    for algo in classifier_algorithms:
        clf = _make_classifier(algo, random_state=random_state, n_jobs=n_jobs)
        y_cls = df_clean[FEASIBILITY_COLUMN].astype(int).to_numpy()
        t0 = time.perf_counter()
        clf.fit(X_all, y_cls)
        dt = time.perf_counter() - t0
        classifiers[algo] = clf
        fit_seconds[(algo, FEASIBILITY_COLUMN)] = dt
        if verbose:
            print(f"  fit {algo:<14s} {FEASIBILITY_COLUMN:<24s} -> {dt:6.2f}s", flush=True)

    training_categories: dict[str, tuple[str, ...]] = {}
    for col in SCENARIO_CATEGORICAL_COLUMNS:
        if col in X_all.columns:
            uniq = X_all[col].astype(str).unique()
            training_categories[col] = tuple(sorted(str(x) for x in uniq))

    return FittedBaselines(
        regressors=regressors,
        joint_mlp=joint_mlp,
        mlp_targets=mlp_targets,
        classifiers=classifiers,
        fit_seconds=fit_seconds,
        training_categories=training_categories,
    )


# ---------------------------------------------------------------------------
# Public API: evaluation
# ---------------------------------------------------------------------------


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        # Sklearn's MAPE clamps the denominator at np.finfo(np.float64).eps
        # so values near zero won't blow up; treat MAPE as approximate
        # for targets that cross or touch zero (energy margin, slope at
        # the feasibility frontier).
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def _classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    # Single-class slice (e.g. one scenario family with 100% feasibility):
    # AUC is undefined; report NaN so the table flags it rather than
    # crashing. Otherwise compute the standard ROC-AUC.
    auc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }


def _per_family_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Yield ``("__all__", df)`` plus one ``(family, df_sub)`` per scenario family."""
    groups: list[tuple[str, pd.DataFrame]] = [("__all__", df)]
    if "scenario_family" in df.columns:
        for fam, sub in df.groupby("scenario_family", observed=True):
            groups.append((str(fam), sub))
    return groups


def evaluate_baselines(
    fitted: FittedBaselines,
    df: pd.DataFrame,
    *,
    split_label: str,
) -> pd.DataFrame:
    """Score every fitted model on ``df``; return a tidy long-format frame.

    Columns: ``algorithm`` ∈ {ridge, random_forest, xgboost, mlp_joint,
    logreg}; ``target``; ``split`` (literal string supplied by caller,
    typically 'val' / 'test'); ``scenario_family`` ('__all__' for the
    aggregate); ``metric`` ∈ {r2, rmse, mape, auc, f1, accuracy,
    positive_rate, n}; ``value``.

    Regression rows live alongside classification rows in the same
    frame; downstream callers filter by ``metric``.
    """
    df_clean = valid_rows(df)
    rows: list[dict[str, Any]] = []

    # --- Regression: per-target estimators -------------------------------
    # Schema v6: feasibility is "not stalled" (positive class flipped).
    feas_mask_full = ~df_clean[FEASIBILITY_COLUMN].astype(bool)
    df_feas = df_clean.loc[feas_mask_full].copy()
    # Guard: empty regression slices (everything in the input split was
    # infeasible after status filtering) skip the regressor block
    # entirely. Returning an empty frame with the right schema is
    # cleaner than letting sklearn raise on a zero-row predict.
    if len(df_feas) == 0:
        regressors_to_score: dict[tuple[str, str], Any] = {}
        X_feas = pd.DataFrame()
    else:
        regressors_to_score = fitted.regressors
        X_feas = build_feature_matrix(df_feas)
    for (algo, target), est in regressors_to_score.items():
        y_pred = np.asarray(est.predict(X_feas))
        for fam, sub in _per_family_groups(df_feas):
            sub_idx = sub.index
            mask = df_feas.index.isin(sub_idx)
            y_true_g = df_feas.loc[mask, target].to_numpy()
            y_pred_g = y_pred[mask]
            if len(y_true_g) < 2:
                continue
            for metric, value in _regression_metrics(y_true_g, y_pred_g).items():
                rows.append(
                    {
                        "algorithm": algo,
                        "target": target,
                        "split": split_label,
                        "scenario_family": fam,
                        "metric": metric,
                        "value": value,
                    }
                )

    # --- Regression: joint MLP -------------------------------------------
    if fitted.joint_mlp is not None and fitted.mlp_targets and len(df_feas) > 0:
        Y_pred_joint = np.asarray(fitted.joint_mlp.predict(X_feas))
        if Y_pred_joint.ndim == 1:
            Y_pred_joint = Y_pred_joint[:, None]
        for j, target in enumerate(fitted.mlp_targets):
            for fam, sub in _per_family_groups(df_feas):
                mask = df_feas.index.isin(sub.index)
                y_true_g = df_feas.loc[mask, target].to_numpy()
                y_pred_g = Y_pred_joint[mask, j]
                if len(y_true_g) < 2:
                    continue
                for metric, value in _regression_metrics(y_true_g, y_pred_g).items():
                    rows.append(
                        {
                            "algorithm": JOINT_MLP_NAME,
                            "target": target,
                            "split": split_label,
                            "scenario_family": fam,
                            "metric": metric,
                            "value": value,
                        }
                    )

    # --- Classification: feasibility -------------------------------------
    if len(df_clean) == 0:
        return pd.DataFrame(rows)
    X_all = build_feature_matrix(df_clean)
    y_true = df_clean[FEASIBILITY_COLUMN].astype(int).to_numpy()
    for algo, clf in fitted.classifiers.items():
        if hasattr(clf, "predict_proba"):
            y_score = np.asarray(clf.predict_proba(X_all))[:, 1]
        else:  # pragma: no cover — every shipped classifier exposes proba
            y_score = np.asarray(clf.predict(X_all)).astype(float)
        for fam, sub in _per_family_groups(df_clean):
            mask = df_clean.index.isin(sub.index)
            y_true_g = y_true[mask]
            y_score_g = y_score[mask]
            if len(y_true_g) < 2:
                continue
            for metric, value in _classification_metrics(y_true_g, y_score_g).items():
                rows.append(
                    {
                        "algorithm": algo,
                        "target": FEASIBILITY_COLUMN,
                        "split": split_label,
                        "scenario_family": fam,
                        "metric": metric,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Acceptance gates (project_plan.md §6 / §7 Layer-1)
# ---------------------------------------------------------------------------

ACCEPTANCE_GATES: dict[str, dict[str, float]] = {
    "range_km": {"r2": 0.95},
    "energy_margin_raw_pct": {"r2": 0.95},
    "slope_capability_deg": {"r2": 0.85},
    "total_mass_kg": {"r2": 0.85},
    FEASIBILITY_COLUMN: {"auc": 0.90},
}
"""Plan-defined Week-6 thresholds keyed by target.

Used by :func:`acceptance_gate` to decide pass/fail per (algorithm,
target) on the test split. A model passes if **all** of its target's
thresholds are met."""


def acceptance_gate(
    metrics_df: pd.DataFrame,
    *,
    split: str = "test",
    family: str = "__all__",
) -> pd.DataFrame:
    """Return one row per (algorithm, target) with pass/fail vs. plan thresholds."""
    sub = metrics_df.query("split == @split and scenario_family == @family")
    out_rows: list[dict[str, Any]] = []
    for (algo, target), grp in sub.groupby(["algorithm", "target"]):
        thresholds = ACCEPTANCE_GATES.get(str(target), {})
        if not thresholds:
            continue
        observed: dict[str, float] = {}
        passes_all = True
        for metric, threshold in thresholds.items():
            row = grp[grp["metric"] == metric]
            if row.empty:
                observed[metric] = float("nan")
                passes_all = False
                continue
            value = float(row["value"].iloc[0])
            observed[metric] = value
            passes_all = passes_all and value >= threshold
        out_rows.append(
            {
                "algorithm": algo,
                "target": target,
                **{f"{k}_observed": v for k, v in observed.items()},
                **{f"{k}_threshold": v for k, v in thresholds.items()},
                "passes": passes_all,
            }
        )
    return pd.DataFrame(out_rows).sort_values(["target", "algorithm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Registry-rover sanity check (Layer-1 from project_plan.md §7)
# ---------------------------------------------------------------------------


def _row_for_registry_rover(
    name: str,
    *,
    training_categories: dict[str, tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a single-row feature DataFrame for a registry rover.

    Returns
    -------
    (X_row, evaluator_metrics)
        ``X_row`` is a 1-row DataFrame with the same column dtypes as
        the LHS training feature matrix (categoricals expressed as
        plain strings — sklearn's OneHotEncoder accepts ``handle_unknown
        ='ignore'`` and XGBoost's ``enable_categorical=True`` happily
        coerces an object column at predict time).

        ``evaluator_metrics`` is the dict of mission metrics the
        deterministic evaluator produces for the same (design, scenario,
        gravity) triple, used as the Layer-1 ground truth.
    """
    from roverdevkit.mission.evaluator import evaluate
    from roverdevkit.validation.rover_registry import registry_by_name

    entry = registry_by_name(name)
    metrics = evaluate(
        entry.design,
        entry.scenario,
        gravity_m_per_s2=entry.gravity_m_per_s2,
        thermal_architecture=entry.thermal_architecture,
    )

    design = entry.design
    scenario = entry.scenario
    row: dict[str, Any] = {
        "design_wheel_radius_m": design.wheel_radius_m,
        "design_wheel_width_m": design.wheel_width_m,
        "design_grouser_height_m": design.grouser_height_m,
        "design_grouser_count": design.grouser_count,
        "design_n_wheels": design.n_wheels,
        "design_chassis_mass_kg": design.chassis_mass_kg,
        "design_wheelbase_m": design.wheelbase_m,
        "design_solar_area_m2": design.solar_area_m2,
        "design_battery_capacity_wh": design.battery_capacity_wh,
        "design_avionics_power_w": design.avionics_power_w,
        "design_peak_wheel_torque_nm": design.peak_wheel_torque_nm,
        "scenario_latitude_deg": scenario.latitude_deg,
        "scenario_mission_duration_earth_days": scenario.mission_duration_earth_days,
        "scenario_max_slope_deg": scenario.max_slope_deg,
        "scenario_operational_duty_cycle": scenario.operational_duty_cycle,
        "scenario_soil_n": float("nan"),  # filled below
        "scenario_soil_k_c": float("nan"),
        "scenario_soil_k_phi": float("nan"),
        "scenario_soil_cohesion_kpa": float("nan"),
        "scenario_soil_friction_angle_deg": float("nan"),
        "scenario_soil_shear_modulus_k_m": float("nan"),
        # Categorical: use the LHS family the rover most resembles so
        # XGBoost native-categorical handling has a value within the
        # learned codebook. Lunar-only registry as of 2026-04-25; the
        # latitude/slope rules below pick the closest LHS family for
        # any new entry.
        "scenario_family": "equatorial_mare_traverse",
        "scenario_terrain_class": str(scenario.terrain_class),
        "scenario_soil_simulant": str(scenario.soil_simulant),
        "scenario_sun_geometry": str(scenario.sun_geometry),
    }

    # Pull the catalogued Bekker parameters for the rover's actual soil
    # so the surrogate sees realistic numeric soil features.
    from roverdevkit.terramechanics.soils import get_soil_parameters

    soil = get_soil_parameters(str(scenario.soil_simulant))
    row["scenario_soil_n"] = soil.n
    row["scenario_soil_k_c"] = soil.k_c
    row["scenario_soil_k_phi"] = soil.k_phi
    row["scenario_soil_cohesion_kpa"] = soil.cohesion_kpa
    row["scenario_soil_friction_angle_deg"] = soil.friction_angle_deg
    row["scenario_soil_shear_modulus_k_m"] = soil.shear_modulus_k_m

    # Pick the family whose latitude band best matches the rover so the
    # categorical encodings line up with how the LHS sampler attached
    # them in training.
    abs_lat = abs(scenario.latitude_deg)
    if abs_lat >= 60.0:
        row["scenario_family"] = "polar_prospecting"
    elif scenario.max_slope_deg >= 18.0:
        row["scenario_family"] = "highland_slope_capability"
    else:
        row["scenario_family"] = "equatorial_mare_traverse"

    X_row = pd.DataFrame([row])
    for col in SCENARIO_CATEGORICAL_COLUMNS:
        if training_categories is not None and col in training_categories:
            # Conform to the training codebook so XGBoost's strict
            # ``enable_categorical=True`` recode doesn't raise. Any
            # unseen value becomes NaN (XGBoost treats it as missing).
            X_row[col] = pd.Categorical(
                X_row[col].astype(str),
                categories=list(training_categories[col]),
            )
        else:
            X_row[col] = X_row[col].astype("category")

    evaluator_metrics: dict[str, Any] = {
        "range_km": metrics.range_km,
        "energy_margin_raw_pct": metrics.energy_margin_raw_pct,
        "slope_capability_deg": metrics.slope_capability_deg,
        "total_mass_kg": metrics.total_mass_kg,
        "stalled": bool(metrics.stalled),
    }
    return X_row, evaluator_metrics


def predict_for_registry_rovers(
    fitted: FittedBaselines,
    *,
    rover_names: tuple[str, ...] = ("Pragyan", "Yutu-2", "MoonRanger", "Rashid-1"),
) -> pd.DataFrame:
    """Layer-1 sanity check: each baseline vs the evaluator on registry rovers.

    Default roster covers two flown lunar rovers (Pragyan, Yutu-2)
    plus two design-target lunar micro-rovers (MoonRanger, Rashid-1).
    The Mars-gravity Sojourner sentinel was removed when the project
    narrowed to lunar micro-rovers (project_log.md 2026-04-25).

    Layer-1 framing
    ---------------
    Output rows carry an ``is_primary`` boolean that splits the targets
    into two groups (``project_log.md`` 2026-04-25 LHS v3 entry):

    - ``is_primary=True`` — :data:`LAYER1_PRIMARY_TARGETS`
      (``total_mass_kg``, ``slope_capability_deg``, ``stalled``).
      Design-axis metrics where the v3 widened LHS bounds put the
      registry inside training support; treated as the real Layer-1
      acceptance set.
    - ``is_primary=False`` — :data:`LAYER1_DIAGNOSTIC_TARGETS`
      (``range_km``, ``energy_margin_raw_pct``). Scenario-OOD because
      the registry's published mission distances are 100-1000x smaller
      than the LHS family budgets; reported for diagnostic purposes
      only and *not* an acceptance signal.

    Columns
    -------
    - ``predicted`` — surrogate output
    - ``evaluator`` — Layer-1 ground truth
    - ``abs_error`` / ``rel_error`` — same convention as the per-scenario
      breakdown so downstream readers can use one mental model.
    - ``is_primary`` — see "Layer-1 framing" above.

    The classifier reports its predicted stall probability
    against the evaluator's binary ``stalled``.
    """
    primary_targets = set(LAYER1_PRIMARY_TARGETS)
    rows: list[dict[str, Any]] = []
    for name in rover_names:
        X_row, evaluator_metrics = _row_for_registry_rover(
            name, training_categories=fitted.training_categories or None
        )
        for (algo, target), est in fitted.regressors.items():
            y_hat = float(np.asarray(est.predict(X_row))[0])
            y_true = float(evaluator_metrics[target])
            rows.append(
                {
                    "rover": name,
                    "algorithm": algo,
                    "target": target,
                    "predicted": y_hat,
                    "evaluator": y_true,
                    "abs_error": y_hat - y_true,
                    "rel_error": (y_hat - y_true) / y_true if y_true != 0 else float("nan"),
                    "is_primary": target in primary_targets,
                }
            )
        if fitted.joint_mlp is not None and fitted.mlp_targets:
            y_hat_vec = np.asarray(fitted.joint_mlp.predict(X_row))
            if y_hat_vec.ndim == 1:
                y_hat_vec = y_hat_vec[None, :]
            for j, target in enumerate(fitted.mlp_targets):
                y_hat = float(y_hat_vec[0, j])
                y_true = float(evaluator_metrics[target])
                rows.append(
                    {
                        "rover": name,
                        "algorithm": JOINT_MLP_NAME,
                        "target": target,
                        "predicted": y_hat,
                        "evaluator": y_true,
                        "abs_error": y_hat - y_true,
                        "rel_error": ((y_hat - y_true) / y_true if y_true != 0 else float("nan")),
                        "is_primary": target in primary_targets,
                    }
                )
        for algo, clf in fitted.classifiers.items():
            if hasattr(clf, "predict_proba"):
                p = float(np.asarray(clf.predict_proba(X_row))[0, 1])
            else:  # pragma: no cover
                p = float(np.asarray(clf.predict(X_row))[0])
            y_true_bool = bool(evaluator_metrics[FEASIBILITY_COLUMN])
            rows.append(
                {
                    "rover": name,
                    "algorithm": algo,
                    "target": FEASIBILITY_COLUMN,
                    "predicted": p,
                    "evaluator": float(y_true_bool),
                    "abs_error": p - float(y_true_bool),
                    "rel_error": float("nan"),
                    "is_primary": FEASIBILITY_COLUMN in primary_targets,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ACCEPTANCE_GATES",
    "CLASSIFIER_ALGORITHMS",
    "FittedBaselines",
    "JOINT_MLP_NAME",
    "LAYER1_DIAGNOSTIC_TARGETS",
    "LAYER1_PRIMARY_TARGETS",
    "REGRESSION_ALGORITHMS",
    "acceptance_gate",
    "evaluate_baselines",
    "fit_baselines",
    "predict_for_registry_rovers",
]
