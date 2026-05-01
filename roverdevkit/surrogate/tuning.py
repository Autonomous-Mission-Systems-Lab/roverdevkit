"""Optuna-based hyperparameter tuning for XGBoost surrogate baselines.

Scope (project_plan.md §6.2 / Week-8 step-3)
--------------------------------------------
Tunes only **XGBoost** — for both the per-target regressors and the
``stalled`` feasibility classifier (schema v6 flipped polarity from
``motor_torque_ok``). The Week-8 step-2 baselines
report (`reports/week8_baselines_v4/SUMMARY.md`) shows XGBoost is within
0.005 R² of the joint MLP on every primary target while being ~7×
faster to fit, which makes it the production candidate for the
Phase 3 NSGA-II constraint loop. Tuning Ridge / RF / LogReg / MLP would
not move the production frontier:

- Ridge is the linear-baseline floor (intentional reference, not a
  production candidate); tuning ``alpha`` won't recover the +0.30 R²
  it loses to non-linear models on energy margin and range.
- Random Forest is already weaker than untuned XGBoost on every
  target; no plausible HP setting closes that gap.
- LogReg already lands at AUC 0.985 on the classifier — saturated;
  a tuned XGBoost is the only candidate that could plausibly edge it.
- MLP is ~7× slower per fit and only a hair better than untuned
  XGB; deferred unless a follow-up experiment specifically calls for it.

Approach
--------
- Sampler: ``TPESampler`` with explicit ``seed`` for reproducibility.
- Objective: held-out **val** R² (regressor) / val AUC (classifier).
  The test split is never seen by the tuner.
- Inside-trial early stopping via XGBoost's ``early_stopping_rounds``
  on the val set (cheaper than optuna pruning callbacks for our
  trial budget).
- Final fit: best params refitted on **train ∪ val**, then scored on
  test. The ``early_stopping_rounds`` is dropped at refit time and
  ``n_estimators`` is fixed to the best-iteration count from the
  tuning run so the refit doesn't extrapolate trees the val set
  never validated.

Returns ``TuningResult`` with the best params, fitted final model,
study summary frame, and timing — enough for downstream code to
serialise the model and report tuned vs untuned in the metrics frame.

This module intentionally does **not** modify ``baselines.py``: the
default-hyperparameter pipeline stays intact so the Week-6 / Week-8
step-2 acceptance numbers remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, roc_auc_score

# Quiet Optuna's per-trial INFO chatter so the CLI driver's own logs stay
# readable. The trial log is still written to the persisted study.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search-space definitions
# ---------------------------------------------------------------------------


def _suggest_xgb_regressor_params(trial: optuna.Trial, *, random_state: int) -> dict[str, Any]:
    """Search space mirrors the Week-6 default config but lets every knob move.

    ``n_estimators`` is allowed up to 1500 with early stopping on the
    val set; the actual count is recovered from ``best_iteration`` when
    refitting on train+val.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": random_state,
    }


def _suggest_xgb_classifier_params(trial: optuna.Trial, *, random_state: int) -> dict[str, Any]:
    """Same axes as the regressor; XGBClassifier accepts the same hyperparameters."""
    return _suggest_xgb_regressor_params(trial, random_state=random_state)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningResult:
    """Bundle of artifacts produced by :func:`tune_xgboost_regressor` /
    :func:`tune_xgboost_classifier`.

    Attributes
    ----------
    target
        Target column name (regressor) or ``stalled``
        (classifier). Used for downstream artifact naming.
    best_params
        Hyperparameter dict applied to the final refit. The
        ``n_estimators`` field is the best-iteration count from
        early stopping rather than the suggested upper bound.
    val_score
        Best objective seen during tuning (val R² for regressors,
        val AUC for classifiers).
    final_model
        XGBoost estimator refitted on train ∪ val with ``best_params``.
        Ready for ``.predict`` / ``.predict_proba`` on the test split.
    n_trials
        Number of completed trials in the study.
    elapsed_seconds
        Wall-clock for the tuning loop (does not include the final
        refit on train+val).
    study_df
        ``study.trials_dataframe()`` — useful for the writeup.
    """

    target: str
    best_params: dict[str, Any]
    val_score: float
    final_model: Any
    n_trials: int
    elapsed_seconds: float
    study_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Public API: regressor tuning
# ---------------------------------------------------------------------------


def tune_xgboost_regressor(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    target: str,
    n_trials: int = 50,
    timeout_seconds: float | None = None,
    random_state: int = 42,
    early_stopping_rounds: int = 25,
    n_jobs: int = -1,
) -> TuningResult:
    """Run a TPE study to maximise val R² for one regression target.

    ``X_*`` may include pandas ``category`` columns — XGBoost handles
    them natively via ``enable_categorical=True`` (set in the trial
    params). ``y_*`` are 1-D arrays of the same length.

    Returns a :class:`TuningResult` whose ``final_model`` was fit on
    ``X_train ∪ X_val`` with the best hyperparameters and the
    early-stopping-best ``n_estimators``.
    """
    import time

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_regressor_params(trial, random_state=random_state)
        model = xgb.XGBRegressor(
            **params,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Early stopping picks the best iteration; use it for scoring
        # so the trial reports the best val R² it actually achieved
        # rather than the round at which trees stopped being added.
        best_iter = int(getattr(model, "best_iteration", params["n_estimators"]))
        trial.set_user_attr("best_iteration", best_iter)
        y_pred = model.predict(X_val)
        return float(r2_score(y_val, y_pred))

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    elapsed = time.perf_counter() - t0

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["n_estimators"] = int(
        best_trial.user_attrs.get("best_iteration", best_params["n_estimators"])
    )
    best_params["tree_method"] = "hist"
    best_params["enable_categorical"] = True
    best_params["random_state"] = random_state

    # Refit on train ∪ val with best params
    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=False)
    y_full = np.concatenate([y_train, y_val])
    final = xgb.XGBRegressor(**best_params, n_jobs=n_jobs)
    final.fit(X_full, y_full)

    return TuningResult(
        target=target,
        best_params=best_params,
        val_score=float(study.best_value),
        final_model=final,
        n_trials=len(study.trials),
        elapsed_seconds=elapsed,
        study_df=study.trials_dataframe(),
    )


# ---------------------------------------------------------------------------
# Public API: classifier tuning
# ---------------------------------------------------------------------------


def tune_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    target: str = "stalled",
    n_trials: int = 50,
    timeout_seconds: float | None = None,
    random_state: int = 42,
    early_stopping_rounds: int = 25,
    n_jobs: int = -1,
) -> TuningResult:
    """Run a TPE study to maximise val AUC on the feasibility classifier.

    Mirrors :func:`tune_xgboost_regressor` but uses :class:`XGBClassifier`
    and ROC-AUC as the objective. ``y_*`` should be ``{0, 1}`` arrays.
    """
    import time

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_classifier_params(trial, random_state=random_state)
        model = xgb.XGBClassifier(
            **params,
            n_jobs=n_jobs,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = int(getattr(model, "best_iteration", params["n_estimators"]))
        trial.set_user_attr("best_iteration", best_iter)
        y_score = model.predict_proba(X_val)[:, 1]
        # Single-class val (no negatives or no positives) makes AUC
        # undefined; the project's v4 splits do not produce this case
        # but the guard is cheap and prevents an obscure crash if a
        # future split rebalances.
        if len(np.unique(y_val)) < 2:
            return float("nan")
        return float(roc_auc_score(y_val, y_score))

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    elapsed = time.perf_counter() - t0

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["n_estimators"] = int(
        best_trial.user_attrs.get("best_iteration", best_params["n_estimators"])
    )
    best_params["tree_method"] = "hist"
    best_params["enable_categorical"] = True
    best_params["random_state"] = random_state

    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=False)
    y_full = np.concatenate([y_train, y_val])
    final = xgb.XGBClassifier(**best_params, n_jobs=n_jobs)
    final.fit(X_full, y_full)

    return TuningResult(
        target=target,
        best_params=best_params,
        val_score=float(study.best_value),
        final_model=final,
        n_trials=len(study.trials),
        elapsed_seconds=elapsed,
        study_df=study.trials_dataframe(),
    )


__all__ = [
    "TuningResult",
    "tune_xgboost_classifier",
    "tune_xgboost_regressor",
]
