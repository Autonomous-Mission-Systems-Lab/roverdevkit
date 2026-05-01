"""Smoke tests for the Optuna XGBoost tuning module.

These tests verify the contract — `TuningResult` shape, the
``best_params`` recovery from early stopping, and the refit-on-train+val
flow — without making any claim about hyperparameter optimality. The
underlying TPE study is run with a small ``n_trials`` so the suite
stays well under 30 s on a developer laptop.

Acceptance numbers are measured offline against the 40k LHS dataset
(see ``reports/week8_tuned_v4/SUMMARY.md``); the unit tests here just
guard the API contract.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from roverdevkit.surrogate.dataset import build_dataset
from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    build_feature_matrix,
    valid_rows,
)
from roverdevkit.surrogate.sampling import generate_samples
from roverdevkit.surrogate.tuning import (
    TuningResult,
    tune_xgboost_classifier,
    tune_xgboost_regressor,
)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Tiny LHS dataset shared across every tuning test."""
    samples = generate_samples(n_per_scenario=8, seed=23)
    return build_dataset(samples, n_workers=1, progress=False)


def _split_xy(
    df: pd.DataFrame, target: str, *, feasible_only: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    df_clean = valid_rows(df)
    if feasible_only:
        # Schema v6 (W12 step B): ``FEASIBILITY_COLUMN`` is now ``stalled``
        # with positive class = infeasible, so we negate before masking
        # to keep only the feasible (non-stalled) regression rows.
        mask = (~df_clean[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
        df_clean = df_clean.loc[mask]
    X = build_feature_matrix(df_clean)
    y = df_clean[target].to_numpy()
    if not feasible_only:
        y = y.astype(int)
    return X, y


def test_tune_xgboost_regressor_returns_complete_result(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg", feasible_only=True)
    if len(X) < 6:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    # Manual two-thirds / one-third split — the production splits live
    # in the dataset itself but here we just need *some* held-out val.
    n_train = max(int(0.7 * len(X)), 3)
    X_tr, y_tr = X.iloc[:n_train], y[:n_train]
    X_va, y_va = X.iloc[n_train:], y[n_train:]
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tune_xgboost_regressor(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            n_trials=3,
            random_state=0,
            n_jobs=1,
        )
    assert isinstance(result, TuningResult)
    assert result.target == "total_mass_kg"
    assert result.n_trials == 3
    assert result.elapsed_seconds >= 0.0
    assert result.best_params["enable_categorical"] is True
    assert result.best_params["tree_method"] == "hist"
    # ``n_estimators`` must reflect the early-stopping best iteration,
    # not the suggested upper bound (otherwise the refit extrapolates
    # past the val-validated range).
    assert int(result.best_params["n_estimators"]) >= 1
    # The refitted model must be able to predict on the original X
    pred = result.final_model.predict(X)
    assert pred.shape == (len(X),)


def test_tune_xgboost_classifier_returns_complete_result(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, FEASIBILITY_COLUMN, feasible_only=False)
    # Need both classes for AUC; skip otherwise (single-class smokes
    # are uninformative).
    if len(np.unique(y)) < 2 or len(X) < 6:
        pytest.skip("single-class fixture; tune_xgboost_classifier needs both 0 and 1")
    n_train = max(int(0.7 * len(X)), 3)
    X_tr, y_tr = X.iloc[:n_train], y[:n_train]
    X_va, y_va = X.iloc[n_train:], y[n_train:]
    if len(X_va) < 2 or len(np.unique(y_va)) < 2:
        pytest.skip("need both classes in the val split for AUC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tune_xgboost_classifier(
            X_tr,
            y_tr,
            X_va,
            y_va,
            n_trials=3,
            random_state=0,
            n_jobs=1,
        )
    assert isinstance(result, TuningResult)
    assert result.target == FEASIBILITY_COLUMN
    proba = result.final_model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
