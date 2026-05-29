"""Calibrated prediction intervals via quantile XGBoost.

Scope
-----
Fits three independent XGBoost quantile regressors per primary
regression target at ``τ ∈ {0.05, 0.50, 0.95}`` to produce point
predictions plus 90 % prediction intervals on top of the corrected
mission evaluator (``data/analytical/lhs_v4.parquet``).

A later pipeline review demoted the
mission-level surrogate from "the headline ML deliverable" to "an
optional acceleration and uncertainty layer." That demotion is the
reason this module exists at all: NSGA-II inner-loop fitness needs a
fast, probabilistic answer; quantile XGB is the cheapest way to get
calibrated PIs without a second UQ family (MC dropout, deep ensembles)
that the methodology paper would not actually use.

Hyperparameter strategy
-----------------------
Each quantile head reuses the tuned-median tuned median hyperparameters
(``reports/tuned_v4/tuned_best_params.json``) — same
``max_depth`` / ``learning_rate`` / regularisation, only the loss
function changes. This is a deliberate choice:

- The tuned-median search already moved the median's HP frontier; ξ-tail
  refits at the same setting are good enough for prediction-interval
  *width* on a smooth, well-sampled corpus like ``lhs_v4``.
- It keeps the writeup honest: the only thing varying across the three
  heads is ``quantile_alpha``, so the empirical coverage delta is
  attributable to the loss, not to per-head HP tuning.
- Re-tuning per quantile would multiply tuning cost by 3 and bias the
  ``τ=0.5`` head away from its tuned-median setting, making the median
  sanity guardrail (``§6 step-4``) less informative.

A future revision can per-tune the tail heads if the empirical 90 %
coverage misses the target in a way that suggests systematic
under/over-confidence. For the v4 dataset that is not currently the
case (see ``reports/intervals_v4/SUMMARY.md``).

Quantile crossings
------------------
Independent quantile regressors can produce ``q05 > q50`` or ``q50 >
q95`` for individual rows. The :meth:`QuantileHeads.predict` API
exposes the raw quantile predictions and a ``crossing_rate`` summary
so the caller can decide whether to repair (e.g. row-wise sort) or
report. Repair via sort is cheap and always non-worse for empirical
coverage; the writeup reports both raw and repaired coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_QUANTILES: tuple[float, float, float] = (0.05, 0.50, 0.95)
"""Default τ levels giving a 90 % central prediction interval.

Picked at the project level because (a) it matches the §7 acceptance
language ("calibrated 90 % prediction intervals") and (b) it is the
common reporting standard in the multi-fidelity / surrogate-UQ
literature. The implementation supports arbitrary triples; only
``calibrate_coverage`` assumes the outer pair is the PI envelope.
"""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantileHeads:
    """Three trained quantile XGBoost regressors for one primary target.

    Attributes
    ----------
    target
        Target column name (e.g. ``range_km``).
    quantiles
        ``(τ_low, τ_mid, τ_hi)`` actually fitted; default 0.05/0.5/0.95.
    models
        Tuple of 3 fitted ``xgb.XGBRegressor`` objects, ordered to
        match :attr:`quantiles`. Each was fit with
        ``objective="reg:quantileerror"`` and the corresponding
        ``quantile_alpha``.
    feature_columns
        Frozen column order the heads were fit on. The
        :meth:`predict` API enforces this so a caller cannot
        accidentally reorder features after training.
    base_params
        Hyperparameters shared across the three heads (everything
        except ``objective`` / ``quantile_alpha``). Persisted so the
        writeup and the saved artifact know exactly which tuned-median
        configuration produced the bundle.
    fit_seconds
        Total wall-clock to fit all three heads (refit on train+val).
    """

    target: str
    quantiles: tuple[float, float, float]
    models: tuple[xgb.XGBRegressor, xgb.XGBRegressor, xgb.XGBRegressor]
    feature_columns: tuple[str, ...]
    base_params: dict[str, Any] = field(default_factory=dict)
    fit_seconds: float = 0.0

    def predict(
        self,
        X: pd.DataFrame,
        *,
        repair_crossings: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return per-quantile predictions for ``X``.

        Parameters
        ----------
        X
            Feature frame. Must contain ``feature_columns`` in any
            order; categoricals must be ``category`` dtype.
        repair_crossings
            If True, row-wise sort the three predictions so the bundle
            never reports ``q_low > q_mid > q_hi`` violations. The
            unrepaired predictions are still recoverable from the
            individual ``models``; this flag controls only what gets
            returned. Default ``False`` so the writeup reports the
            raw model output.
        """
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise KeyError(f"X is missing required columns: {missing}")
        X_aligned = X[list(self.feature_columns)]
        preds = np.column_stack(
            [np.asarray(m.predict(X_aligned)) for m in self.models]
        )  # shape (N, 3)
        if repair_crossings:
            preds = np.sort(preds, axis=1)
        keys = (f"q{int(round(q * 100)):02d}" for q in self.quantiles)
        return {k: preds[:, i] for i, k in enumerate(keys)}

    def crossing_rate(self, X: pd.DataFrame) -> float:
        """Fraction of rows where the raw quantile triple is non-monotone."""
        preds = self.predict(X, repair_crossings=False)
        keys = list(preds.keys())
        a, b, c = preds[keys[0]], preds[keys[1]], preds[keys[2]]
        bad = (a > b) | (b > c) | (a > c)
        return float(bad.mean())

    def save(self, path: Path) -> None:
        """Serialise via joblib. Models are picklable XGBoost regressors."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> QuantileHeads:
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__} at {path}, got {type(obj)!r}")
        return obj


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_quantile_heads(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    target: str,
    base_params: dict[str, Any],
    quantiles: tuple[float, float, float] = DEFAULT_QUANTILES,
    early_stopping_rounds: int = 25,
    n_jobs: int = -1,
) -> QuantileHeads:
    """Fit three quantile XGBoost regressors sharing ``base_params``.

    Parameters
    ----------
    X_train / y_train / X_val / y_val
        Same train / val split used by tuned-median. Validation drives
        early stopping; the final refit happens on ``train ∪ val``
        with the early-stopping-best ``n_estimators`` per head.
    target
        Target name (used only for downstream artifact naming).
    base_params
        Output of :func:`json.load` on
        ``reports/tuned_v4/tuned_best_params.json[target]`` — a
        dict containing ``n_estimators``, ``max_depth``,
        ``learning_rate``, ``subsample``, ``colsample_bytree``,
        ``min_child_weight``, ``reg_alpha``, ``reg_lambda``,
        ``gamma``, ``tree_method``, ``enable_categorical``,
        ``random_state``. ``objective`` and ``quantile_alpha`` are
        injected per-head and override anything in ``base_params``.
    quantiles
        Triple of τ values. Default ``(0.05, 0.5, 0.95)``.
    early_stopping_rounds
        Patience on the val pinball loss. Mirrors tuned-median tuning.
    n_jobs
        Plumbed through to XGBoost.

    Notes
    -----
    Quantile regression in XGBoost (≥ 2.0) uses
    ``objective="reg:quantileerror"`` plus ``quantile_alpha=τ``. The
    default eval metric is the pinball loss at ``τ``, which is what
    drives early stopping here.
    """
    import time

    feature_columns = tuple(str(c) for c in X_train.columns)
    if tuple(str(c) for c in X_val.columns) != feature_columns:
        raise ValueError("X_train and X_val must share the same column order")

    # Keep base_params clean: drop anything quantile-specific so we
    # are the sole authority on it.
    shared = {k: v for k, v in base_params.items() if k not in ("objective", "quantile_alpha")}

    models: list[xgb.XGBRegressor] = []
    t0 = time.perf_counter()
    for tau in quantiles:
        m = xgb.XGBRegressor(
            **shared,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            objective="reg:quantileerror",
            quantile_alpha=float(tau),
        )
        m.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Refit on train ∪ val with the early-stopping-best
        # n_estimators so the deployed head matches what we saw at
        # validation time. We drop early_stopping for the refit
        # (no held-out set to monitor on the combined data).
        best_iter = int(getattr(m, "best_iteration", shared["n_estimators"]))
        refit_params = dict(shared)
        refit_params["n_estimators"] = max(1, best_iter + 1)
        final = xgb.XGBRegressor(
            **refit_params,
            n_jobs=n_jobs,
            objective="reg:quantileerror",
            quantile_alpha=float(tau),
        )
        X_full = pd.concat([X_train, X_val], axis=0, ignore_index=False)
        y_full = np.concatenate([y_train, y_val])
        final.fit(X_full, y_full)
        models.append(final)

    elapsed = time.perf_counter() - t0
    return QuantileHeads(
        target=target,
        quantiles=tuple(float(q) for q in quantiles),  # type: ignore[arg-type]
        models=(models[0], models[1], models[2]),
        feature_columns=feature_columns,
        base_params=dict(shared),
        fit_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Coverage calibration
# ---------------------------------------------------------------------------


def coverage_table(
    bundle: QuantileHeads,
    X: pd.DataFrame,
    y_true: np.ndarray,
    *,
    scenario_family: pd.Series | None = None,
    repair_crossings: bool = False,
) -> pd.DataFrame:
    """Empirical coverage of the outer quantile pair as a PI.

    Returns a long-format frame with rows for the overall split and
    one row per ``scenario_family`` value (if provided). Columns:

    - ``target`` — propagated from the bundle.
    - ``scenario_family`` — ``__all__`` for the overall row.
    - ``n`` — sample count in the cell.
    - ``nominal`` — 1 − (τ_hi − τ_lo); for the default 0.05/0.95
      triple this is 0.90.
    - ``empirical`` — fraction of rows with ``q_low ≤ y ≤ q_hi``.
    - ``mean_width`` — mean of ``q_hi − q_low`` (units of the target).
    - ``median_width`` — median of ``q_hi − q_low``.
    - ``crossing_rate`` — fraction of rows where the raw quantile
      triple is non-monotone (only meaningful when
      ``repair_crossings=False``).
    """
    if scenario_family is not None and len(scenario_family) != len(X):
        raise ValueError("scenario_family must have the same length as X")
    preds = bundle.predict(X, repair_crossings=repair_crossings)
    keys = list(preds.keys())  # ordered: q_lo, q_mid, q_hi
    q_lo, q_hi = preds[keys[0]], preds[keys[-1]]
    nominal = float(bundle.quantiles[-1] - bundle.quantiles[0])
    inside = (y_true >= q_lo) & (y_true <= q_hi)
    width = q_hi - q_lo
    raw_preds = bundle.predict(X, repair_crossings=False)
    raw_lo, raw_mid, raw_hi = (
        raw_preds[keys[0]],
        raw_preds[keys[1]],
        raw_preds[keys[-1]],
    )
    crossings = (raw_lo > raw_mid) | (raw_mid > raw_hi) | (raw_lo > raw_hi)

    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, np.ndarray]] = [("__all__", np.ones(len(X), dtype=bool))]
    if scenario_family is not None:
        for fam in sorted(scenario_family.dropna().unique()):
            mask = (scenario_family == fam).to_numpy()
            groups.append((str(fam), mask))
    for fam, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "target": bundle.target,
                "scenario_family": fam,
                "n": n,
                "nominal": nominal,
                "empirical": float(inside[mask].mean()),
                "mean_width": float(width[mask].mean()),
                "median_width": float(np.median(width[mask])),
                "crossing_rate": float(crossings[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_QUANTILES",
    "QuantileHeads",
    "coverage_table",
    "fit_quantile_heads",
]
