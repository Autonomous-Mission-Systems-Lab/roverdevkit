"""Smoke tests for quantile-XGBoost prediction-interval calibration.

These tests verify the contract — :class:`QuantileHeads` shape,
:meth:`predict` enforcing the feature-column order, save/load
round-trip, the coverage-table schema — without making any claim
about empirical 90 % coverage. The full coverage numbers are measured
offline against the 40k LHS dataset and live in
``reports/week8_intervals_v4/SUMMARY.md``.

The fixture is identical to ``test_surrogate_tuning.py`` (and so is
the (X, y) split helper) so the suite runtime stays well under 10 s.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
from roverdevkit.surrogate.uncertainty import (
    DEFAULT_QUANTILES,
    QuantileHeads,
    coverage_table,
    fit_quantile_heads,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """Tiny LHS dataset shared across every uncertainty test."""
    samples = generate_samples(n_per_scenario=8, seed=23)
    return build_dataset(samples, n_workers=1, progress=False)


def _split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    df_clean = valid_rows(df)
    # Schema v6 (W12 step B): ``FEASIBILITY_COLUMN`` is now ``stalled``
    # with positive class = infeasible (the failure mode), so we negate
    # before masking to keep only the *feasible* (non-stalled) rows the
    # quantile heads need.
    mask = (~df_clean[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
    df_clean = df_clean.loc[mask]
    X = build_feature_matrix(df_clean).reset_index(drop=True)
    y = df_clean[target].to_numpy()
    return X, y


def _split_train_val(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Two-thirds / one-third deterministic split for the smokes."""
    n_train = max(int(0.7 * len(X)), 3)
    return X.iloc[:n_train], y[:n_train], X.iloc[n_train:], y[n_train:]


def _tiny_base_params() -> dict:
    """Mirror W8 step-3 schema with cheap values so the smoke runs fast."""
    return {
        "n_estimators": 60,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fit_quantile_heads_returns_complete_bundle(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    assert isinstance(bundle, QuantileHeads)
    assert bundle.target == "total_mass_kg"
    assert bundle.quantiles == DEFAULT_QUANTILES
    assert len(bundle.models) == 3
    assert bundle.feature_columns == tuple(X_tr.columns.astype(str))
    assert bundle.fit_seconds >= 0.0


def test_predict_returns_quantile_keyed_dict(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    preds = bundle.predict(X)
    assert set(preds.keys()) == {"q05", "q50", "q95"}
    for arr in preds.values():
        assert arr.shape == (len(X),)
        assert np.all(np.isfinite(arr))


def test_predict_repair_crossings_is_monotone(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    preds = bundle.predict(X, repair_crossings=True)
    assert np.all(preds["q05"] <= preds["q50"])
    assert np.all(preds["q50"] <= preds["q95"])


def test_predict_rejects_missing_columns(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    bad = X.drop(columns=[X.columns[0]])
    with pytest.raises(KeyError):
        bundle.predict(bad)


def test_coverage_table_schema(small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    df_clean = valid_rows(small_df)
    # Schema v6: keep only the feasible (non-``stalled``) rows.
    df_clean = df_clean.loc[~df_clean[FEASIBILITY_COLUMN].astype(bool)]
    fam = df_clean["scenario_family"].astype(str).reset_index(drop=True)
    cov = coverage_table(bundle, X, y, scenario_family=fam, repair_crossings=False)
    expected_cols = {
        "target",
        "scenario_family",
        "n",
        "nominal",
        "empirical",
        "mean_width",
        "median_width",
        "crossing_rate",
    }
    assert expected_cols.issubset(cov.columns)
    assert (cov["target"] == "total_mass_kg").all()
    np.testing.assert_allclose(cov["nominal"].to_numpy(), 0.90)
    overall = cov.query("scenario_family == '__all__'")
    assert len(overall) == 1
    assert 0.0 <= float(overall["empirical"].iloc[0]) <= 1.0


def test_save_load_roundtrip(tmp_path: Path, small_df: pd.DataFrame) -> None:
    X, y = _split_xy(small_df, "total_mass_kg")
    if len(X) < 8:
        pytest.skip("LHS happened to land too few feasible rows for the smoke test")
    X_tr, y_tr, X_va, y_va = _split_train_val(X, y)
    if len(X_va) < 2:
        pytest.skip("not enough rows for a held-out val split in this fixture")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target="total_mass_kg",
            base_params=_tiny_base_params(),
            n_jobs=1,
        )

    path = tmp_path / "bundle.joblib"
    bundle.save(path)
    loaded = QuantileHeads.load(path)

    assert loaded.target == bundle.target
    assert loaded.quantiles == bundle.quantiles
    assert loaded.feature_columns == bundle.feature_columns
    np.testing.assert_allclose(
        bundle.predict(X)["q50"],
        loaded.predict(X)["q50"],
    )
