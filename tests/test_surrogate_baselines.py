"""Unit tests for the baseline-surrogate baseline surrogate models.

These tests focus on **shape and contract**, not on accuracy:

- ``fit_baselines`` produces the expected number of regressors / one
  classifier per algorithm / a single joint MLP keyed on the right
  targets.
- ``evaluate_baselines`` returns a tidy long-format frame with the
  expected ``(algorithm, target, split, scenario_family, metric,
  value)`` schema and at least one ``__all__`` row per (algorithm,
  target, metric) cell.
- ``acceptance_gate`` returns one row per ``(algorithm, target)`` with
  a boolean ``passes`` column.
- ``predict_for_registry_rovers`` produces one row per ``(rover,
  algorithm, target)`` and survives a registry-rover whose categorical
  values may not appear in the small training set (the categorical
  conform path).

A single small in-memory dataset (``n_per_scenario=8`` -> 32 rows) is
shared across all tests via a module-scoped fixture so the evaluator
is only invoked once. This dataset is too small to hit the baseline-surrogate R²
gates, which is intentional: accuracy is measured offline against the
40k LHS dataset, not in unit tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

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
from roverdevkit.surrogate.dataset import build_dataset
from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    PRIMARY_REGRESSION_TARGETS,
)
from roverdevkit.surrogate.sampling import generate_samples

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_baseline_df() -> pd.DataFrame:
    """Tiny but real-schema training/test set; one evaluator run per row."""
    samples = generate_samples(n_per_scenario=8, seed=17)
    return build_dataset(samples, n_workers=1, progress=False)


@pytest.fixture(scope="module")
def fitted(small_baseline_df: pd.DataFrame) -> FittedBaselines:
    """Fit every baseline once and reuse across the module."""
    train_df = small_baseline_df[small_baseline_df["split"] == "train"]
    if len(train_df) == 0:
        # Very small datasets sometimes leave the train slot empty if
        # the LHS happens to assign all rows to val/test. Fall back to
        # using the whole dataset.
        train_df = small_baseline_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_baselines(
            train_df,
            fit_mlp=True,
            n_jobs=1,
            random_state=42,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# fit_baselines
# ---------------------------------------------------------------------------


def test_fit_baselines_produces_one_regressor_per_target_per_algorithm(
    fitted: FittedBaselines,
) -> None:
    expected = {(algo, t) for algo in REGRESSION_ALGORITHMS for t in PRIMARY_REGRESSION_TARGETS}
    assert set(fitted.regressors.keys()) == expected


def test_fit_baselines_attaches_joint_mlp(fitted: FittedBaselines) -> None:
    assert fitted.joint_mlp is not None
    assert tuple(fitted.mlp_targets) == tuple(PRIMARY_REGRESSION_TARGETS)


def test_fit_baselines_produces_one_classifier_per_algorithm(
    fitted: FittedBaselines,
) -> None:
    assert set(fitted.classifiers.keys()) == set(CLASSIFIER_ALGORITHMS)


def test_fit_baselines_records_training_categories(fitted: FittedBaselines) -> None:
    """The conform path needs a non-empty codebook for every cat column."""
    expected_cols = {
        "scenario_family",
        "scenario_terrain_class",
        "scenario_soil_simulant",
        "scenario_sun_geometry",
    }
    assert set(fitted.training_categories.keys()) == expected_cols
    for col, levels in fitted.training_categories.items():
        assert len(levels) >= 1, f"empty codebook for {col}"
        assert all(isinstance(v, str) for v in levels)


def test_fit_baselines_records_per_fit_wallclock(fitted: FittedBaselines) -> None:
    for key in fitted.regressors:
        assert key in fitted.fit_seconds
        assert fitted.fit_seconds[key] >= 0.0
    assert (JOINT_MLP_NAME, "joint") in fitted.fit_seconds


def test_fit_baselines_skips_mlp_when_disabled(small_baseline_df: pd.DataFrame) -> None:
    train_df = small_baseline_df[small_baseline_df["split"] == "train"]
    if len(train_df) == 0:
        train_df = small_baseline_df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_baselines(train_df, fit_mlp=False, n_jobs=1, verbose=False)
    assert out.joint_mlp is None
    assert out.mlp_targets == ()


# ---------------------------------------------------------------------------
# evaluate_baselines
# ---------------------------------------------------------------------------


def test_evaluate_baselines_returns_tidy_long_frame(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    # Use the full small df so we're guaranteed at least one feasible
    # row and at least one infeasible row regardless of how the LHS
    # split fractions land at this scale.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="test")
    assert set(m.columns) == {
        "algorithm",
        "target",
        "split",
        "scenario_family",
        "metric",
        "value",
    }
    assert (m["split"] == "test").all()
    assert m["value"].dtype.kind in {"f", "i"}


def test_evaluate_baselines_covers_every_algorithm_target_pair(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    """Every (algo, target) cell appears at least once in the __all__ slice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="all")
    overall = m[m["scenario_family"] == "__all__"]

    seen_pairs = {(row["algorithm"], row["target"]) for _, row in overall.iterrows()}
    expected_pairs: set[tuple[str, str]] = set()
    for algo in REGRESSION_ALGORITHMS:
        for t in PRIMARY_REGRESSION_TARGETS:
            expected_pairs.add((algo, t))
    for t in PRIMARY_REGRESSION_TARGETS:
        expected_pairs.add((JOINT_MLP_NAME, t))
    for algo in CLASSIFIER_ALGORITHMS:
        expected_pairs.add((algo, FEASIBILITY_COLUMN))
    assert expected_pairs.issubset(seen_pairs)


def test_evaluate_baselines_emits_per_scenario_breakdown(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    """At least one scenario family appears beside ``__all__``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="all")
    families = set(m["scenario_family"].unique())
    families.discard("__all__")
    assert len(families) >= 1, "expected at least one per-family slice in the eval frame"


def test_evaluate_baselines_classification_metrics_present(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="all")
    cls_rows = m[m["target"] == FEASIBILITY_COLUMN]
    metrics = set(cls_rows["metric"].unique())
    assert {"auc", "f1", "accuracy", "n", "positive_rate"}.issubset(metrics)


def test_evaluate_baselines_regression_metrics_present(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="all")
    reg_rows = m[m["target"] == "range_km"]
    metrics = set(reg_rows["metric"].unique())
    assert {"r2", "rmse", "mape", "n"}.issubset(metrics)


# ---------------------------------------------------------------------------
# acceptance_gate
# ---------------------------------------------------------------------------


def test_acceptance_gate_one_row_per_algorithm_target(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    # Use the full small df so the test/__all__ slice has feasible rows.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="test")
    g = acceptance_gate(m, split="test")

    assert "passes" in g.columns
    assert g["passes"].dtype == bool

    seen = set(zip(g["algorithm"], g["target"], strict=False))
    for target in PRIMARY_REGRESSION_TARGETS:
        for algo in (*REGRESSION_ALGORITHMS, JOINT_MLP_NAME):
            assert (algo, target) in seen, f"missing acceptance row for {algo}/{target}"


def test_acceptance_gate_targets_match_plan_thresholds(
    fitted: FittedBaselines, small_baseline_df: pd.DataFrame
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate_baselines(fitted, small_baseline_df, split_label="test")
    g = acceptance_gate(m, split="test", family="__all__")
    gated_targets = set(g["target"].unique())
    plan_targets = {
        t for t in ACCEPTANCE_GATES if t in PRIMARY_REGRESSION_TARGETS or t == FEASIBILITY_COLUMN
    }
    assert plan_targets.issubset(gated_targets)


# ---------------------------------------------------------------------------
# predict_for_registry_rovers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_predict_for_registry_rovers_schema(fitted: FittedBaselines) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = predict_for_registry_rovers(fitted)

    assert set(df.columns) == {
        "rover",
        "algorithm",
        "target",
        "predicted",
        "evaluator",
        "abs_error",
        "rel_error",
        "is_primary",
    }
    assert set(df["rover"]) == {"Pragyan", "Yutu-2", "MoonRanger", "Rashid-1"}

    # is_primary partitions targets into design-axis vs scenario-OOD groups
    # (see baselines.LAYER1_PRIMARY_TARGETS / LAYER1_DIAGNOSTIC_TARGETS).
    primary_targets = set(df.loc[df["is_primary"], "target"])
    diagnostic_targets = set(df.loc[~df["is_primary"], "target"])
    assert primary_targets == {
        "total_mass_kg",
        "slope_capability_deg",
        "stalled",
    }
    assert diagnostic_targets == {"range_km", "energy_margin_raw_pct"}

    # Every rover should have one row per regression (algo, target) cell
    # plus the joint MLP plus the classifiers; no NaN in evaluator/predicted.
    assert df["predicted"].notna().all()
    assert df["evaluator"].notna().all()


@pytest.mark.slow
def test_predict_for_registry_rovers_handles_unseen_categories(
    fitted: FittedBaselines,
) -> None:
    """The ``training_categories`` codebook conforms unseen levels to NaN.

    With the very small ``n_per_scenario=8`` fixture the LHS sampler is
    unlikely to have hit every catalogued soil simulant, so at least
    one registry rover almost certainly hits the conform path. This
    test asserts the call returns finite predictions rather than
    raising the XGBoost strict-recode error.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = predict_for_registry_rovers(fitted)
    assert np.isfinite(df["predicted"].to_numpy()).all()
