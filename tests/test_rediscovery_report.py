"""Tests for the rediscovery LOO orchestration and artifact writer.

Three groups:

1. **Failure capture.** A per-rover RuntimeError does not abort the
   sweep; it lands in ``RediscoveryRunSummary.failures``.
2. **Aggregation.** ``summarize_results`` produces the expected columns
   and per-rover content from a real (smoke-budget) sweep.
3. **Artifact writer.** ``write_loo_artifacts`` emits the documented
   file set and the markdown rollup contains the methodology + the
   per-rover table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from roverdevkit.validation.rediscovery_report import (
    DEFAULT_PER_ROVER_OVERRIDES,
    RediscoveryRunSummary,
    run_rediscovery_loo,
    summarize_results,
    write_loo_artifacts,
)
from roverdevkit.validation.rover_rediscovery import (
    rediscover_all,
    rediscover_ensemble,
)


# ---------------------------------------------------------------------------
# A smoke-budget LOO sweep cached once for the whole module.
#
# Only Pragyan is included (subset via per_rover_overrides on every
# *other* rover... actually we just restrict via flown_only and the
# default flown registry happens to be Pragyan + Yutu-2). For test
# headroom we widen mass_ceiling_slop on both to 0.20 and run at
# pop=24 / gen=4 - same budget used in test_rover_rediscovery.py.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loo_smoke_summary() -> RediscoveryRunSummary:
    return run_rediscovery_loo(
        flown_only=True,
        seed=0,
        default_population_size=24,
        default_n_generations=4,
        default_mass_ceiling_slop=0.20,
        per_rover_overrides={},
    )


# ---------------------------------------------------------------------------
# Group 0: defaults and rediscover_all override plumbing
# ---------------------------------------------------------------------------


def test_default_per_rover_overrides_records_cadre_budget() -> None:
    """CADRE-unit has documented overrides for ultra-micro feasibility."""
    assert "CADRE-unit" in DEFAULT_PER_ROVER_OVERRIDES
    cadre = DEFAULT_PER_ROVER_OVERRIDES["CADRE-unit"]
    assert cadre["population_size"] == 80
    assert cadre["n_generations"] == 12
    assert cadre["mass_ceiling_slop"] == 0.50


def test_rediscover_all_rejects_unknown_override_keys() -> None:
    """rediscover_all guards against typo-driven silent param drops."""
    with pytest.raises(KeyError, match="unknown keys"):
        rediscover_all(
            flown_only=True,
            population_size=8,
            n_generations=2,
            per_rover_overrides={"Pragyan": {"nonsense_param": 1}},
        )


# ---------------------------------------------------------------------------
# Group 1: failure capture
# ---------------------------------------------------------------------------


def test_failure_capture_does_not_abort_sweep() -> None:
    """An impossible mass-ceiling forces a per-rover RuntimeError; the
    sweep should keep going and land the failure in ``failures``.

    We force the failure by setting ``mass_ceiling_slop = -0.99`` (mass
    budget at 1 % of modelled mass; every individual is infeasible).
    """
    summary = run_rediscovery_loo(
        flown_only=True,
        seed=0,
        default_population_size=8,
        default_n_generations=2,
        default_mass_ceiling_slop=-0.99,
        per_rover_overrides={},
    )
    assert summary.results == []
    assert set(summary.failures) == {"Pragyan", "Yutu-2"}
    assert not summary.all_succeeded


def test_failure_summary_preserves_kwargs_snapshot() -> None:
    summary = run_rediscovery_loo(
        flown_only=True,
        seed=0,
        default_population_size=8,
        default_n_generations=2,
        default_mass_ceiling_slop=-0.99,
        per_rover_overrides={},
    )
    assert summary.default_kwargs == {
        "population_size": 8,
        "n_generations": 2,
        "mass_ceiling_slop": -0.99,
        "seed": 0,
        "n_seeds": 1,
        "backend": "evaluator",
        "evaluator_eval_cap": 1000,
    }
    assert summary.per_rover_overrides == {}


# ---------------------------------------------------------------------------
# Group 2: aggregation
# ---------------------------------------------------------------------------


_EXPECTED_SUMMARY_COLUMNS: set[str] = {
    "rover_name",
    "is_flown",
    "class_generic_scenario",
    "mass_modelled_kg",
    "mass_budget_kg",
    "pareto_front_size",
    "design_space_distance",
    "pareto_dominated",
    "abs_err_median_pct",
    "abs_err_max_pct",
    "abs_err_max_var",
    "n_wheels_matches",
    "grouser_count_matches",
    "population_size",
    "n_generations",
    "mass_ceiling_slop",
}


def test_summarize_results_columns(loo_smoke_summary: RediscoveryRunSummary) -> None:
    df = summarize_results(loo_smoke_summary)
    assert set(df.columns) == _EXPECTED_SUMMARY_COLUMNS


def test_summarize_results_one_row_per_success(
    loo_smoke_summary: RediscoveryRunSummary,
) -> None:
    df = summarize_results(loo_smoke_summary)
    assert len(df) == len(loo_smoke_summary.results)
    assert set(df["rover_name"]) == {r.rover_name for r in loo_smoke_summary.results}


def test_summarize_results_design_space_distance_nonneg(
    loo_smoke_summary: RediscoveryRunSummary,
) -> None:
    df = summarize_results(loo_smoke_summary)
    assert (df["design_space_distance"] >= 0.0).all()


def test_summarize_results_records_per_rover_budget(
    loo_smoke_summary: RediscoveryRunSummary,
) -> None:
    df = summarize_results(loo_smoke_summary)
    assert (df["population_size"] == 24).all()
    assert (df["n_generations"] == 4).all()
    for slop in df["mass_ceiling_slop"]:
        assert slop == pytest.approx(0.20)


def test_summarize_results_abs_err_max_var_is_a_real_variable(
    loo_smoke_summary: RediscoveryRunSummary,
) -> None:
    valid_vars = {
        "wheel_radius_m",
        "wheel_width_m",
        "grouser_height_m",
        "chassis_mass_kg",
        "wheelbase_m",
        "solar_area_m2",
        "battery_capacity_wh",
        "avionics_power_w",
        "peak_wheel_torque_nm",
    }
    df = summarize_results(loo_smoke_summary)
    assert set(df["abs_err_max_var"]) <= valid_vars


# ---------------------------------------------------------------------------
# Group 3: artifact writer
# ---------------------------------------------------------------------------


def test_write_loo_artifacts_emits_documented_files(
    loo_smoke_summary: RediscoveryRunSummary,
    tmp_path: Path,
) -> None:
    written = write_loo_artifacts(loo_smoke_summary, tmp_path)
    assert "summary" in written
    assert "failures" in written
    assert "report" in written
    for r in loo_smoke_summary.results:
        slug = r.rover_name.lower().replace("-", "_")
        assert slug in written
    for name, path in written.items():
        assert path.exists(), f"missing artifact {name}: {path}"


def test_write_loo_artifacts_csv_loads_as_dataframe(
    loo_smoke_summary: RediscoveryRunSummary,
    tmp_path: Path,
) -> None:
    written = write_loo_artifacts(loo_smoke_summary, tmp_path)
    df = pd.read_csv(written["summary"])
    assert set(df.columns) == _EXPECTED_SUMMARY_COLUMNS
    assert len(df) == len(loo_smoke_summary.results)


def test_write_loo_artifacts_per_rover_json_round_trips(
    loo_smoke_summary: RediscoveryRunSummary,
    tmp_path: Path,
) -> None:
    """Every per-rover JSON loads back as a dict with the expected keys."""
    written = write_loo_artifacts(loo_smoke_summary, tmp_path)
    for r in loo_smoke_summary.results:
        slug = r.rover_name.lower().replace("-", "_")
        payload = json.loads(written[slug].read_text())
        assert payload["rover_name"] == r.rover_name
        assert payload["class_generic_scenario"] == r.class_generic_scenario
        assert payload["design_space_distance"] == pytest.approx(r.design_space_distance)
        assert "pareto_front" in payload
        assert len(payload["pareto_front"]) == len(
            r.optimization_result.design_vectors
        )


def test_write_loo_artifacts_failures_json_always_written(
    loo_smoke_summary: RediscoveryRunSummary,
    tmp_path: Path,
) -> None:
    """failures.json exists even when every rover succeeded."""
    written = write_loo_artifacts(loo_smoke_summary, tmp_path)
    failures = json.loads(written["failures"].read_text())
    assert failures == loo_smoke_summary.failures


def test_write_loo_artifacts_markdown_has_methodology_section(
    loo_smoke_summary: RediscoveryRunSummary,
    tmp_path: Path,
) -> None:
    written = write_loo_artifacts(loo_smoke_summary, tmp_path)
    md = written["report"].read_text()
    assert "Layer-5 rediscovery validation" in md
    assert "Methodology" in md
    assert "Per-rover results" in md
    if loo_smoke_summary.results:
        assert "Aggregate statistics" in md
        for r in loo_smoke_summary.results:
            assert r.rover_name in md


# ---------------------------------------------------------------------------
# Group 4: rediscover_ensemble and ensemble-aware run_rediscovery_loo
# ---------------------------------------------------------------------------


def test_rediscover_ensemble_merges_seeds_and_tightens_distance() -> None:
    """Union of N seeds' fronts has min distance <= any single seed's min."""
    single = rediscover_ensemble(
        "Pragyan",
        population_size=24,
        n_generations=4,
        mass_ceiling_slop=0.20,
        n_seeds=1,
        base_seed=0,
        evaluator_eval_cap=200,
    )
    ensemble = rediscover_ensemble(
        "Pragyan",
        population_size=24,
        n_generations=4,
        mass_ceiling_slop=0.20,
        n_seeds=3,
        base_seed=0,
        evaluator_eval_cap=200,
    )
    assert ensemble.design_space_distance <= single.design_space_distance + 1e-9
    assert len(ensemble.optimization_result.design_vectors) >= len(
        single.optimization_result.design_vectors
    )


def test_rediscover_ensemble_rejects_zero_seeds() -> None:
    with pytest.raises(ValueError, match="n_seeds"):
        rediscover_ensemble(
            "Pragyan",
            n_seeds=0,
            population_size=8,
            n_generations=2,
            evaluator_eval_cap=200,
        )


def test_rediscover_ensemble_propagates_total_failure() -> None:
    """When every seed fails, raise a RuntimeError that surfaces the last reason."""
    with pytest.raises(RuntimeError, match="every NSGA-II seed failed"):
        rediscover_ensemble(
            "Pragyan",
            n_seeds=2,
            population_size=8,
            n_generations=2,
            mass_ceiling_slop=-0.99,
            evaluator_eval_cap=200,
        )


def test_run_rediscovery_loo_routes_through_ensemble_when_multi_seed() -> None:
    """n_seeds > 1 should land in summary.default_kwargs and the merged front."""
    summary = run_rediscovery_loo(
        flown_only=True,
        seed=0,
        default_population_size=16,
        default_n_generations=3,
        default_mass_ceiling_slop=0.20,
        per_rover_overrides={},
        n_seeds=2,
    )
    assert summary.default_kwargs["n_seeds"] == 2
    assert summary.default_kwargs["backend"] == "evaluator"
    # Ensemble fronts are the concatenated union; with two seeds we should
    # see roughly double the population's worth of points compared with a
    # single seed at the same hyperparameters, modulo Pareto filtering
    # inside each seed.
    for r in summary.results:
        assert len(r.optimization_result.design_vectors) >= 1


def test_run_rediscovery_loo_surrogate_backend_requires_bundles() -> None:
    """The surrogate backend without bundles is a programmer error, not a
    per-rover feasibility failure. The constructor's ValueError must
    propagate (not be silently swallowed into ``summary.failures``)."""
    with pytest.raises(ValueError, match="surrogate backend requires"):
        run_rediscovery_loo(
            flown_only=True,
            seed=0,
            default_population_size=16,
            default_n_generations=3,
            default_mass_ceiling_slop=0.20,
            per_rover_overrides={},
            backend="surrogate",
            bundles=None,
        )
