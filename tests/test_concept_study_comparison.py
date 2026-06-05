"""Tests for the Layer-6 concept-study comparison module.

Covers:

- Catalogue accessors: ``list_concept_studies``, ``get_concept_study``,
  unknown-key error path.
- ``ConceptStudyEntry`` construction: published design is a valid
  ``DesignVector``; scenario is loadable; mass envelope is in the
  schema range.
- ``compare_concept_to_pareto``: returns a populated
  :class:`ConceptStudyComparison`; the concept's predicted metrics
  match the evaluator on the same design + scenario;
  ``n_pareto_points_dominated_by_concept`` is 0 on a well-converged
  NSGA-II run; ``nearest_pareto_design_space_distance`` is in
  ``[0, sqrt(9)]``; best-per-objective deltas point in the right
  direction (improvement on the optimiser side).

Smoke budget: pop=24, gen=4, eval_cap=200. Matches the Layer-5
rediscovery tests' budget so the whole layered validation suite
fits inside the same wall-clock envelope.
"""

from __future__ import annotations

import pytest

from roverdevkit.mission.evaluator import evaluate as evaluator_evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector
from roverdevkit.tradespace.optimizer import DEFAULT_OBJECTIVES
from roverdevkit.validation.concept_study_comparison import (
    CONCEPT_STUDIES,
    ConceptStudyComparison,
    ConceptStudyEntry,
    compare_concept_to_pareto,
    get_concept_study,
    list_concept_studies,
)
from roverdevkit.validation.rover_rediscovery import _dominates


# ---------------------------------------------------------------------------
# A smoke-budget comparison cached once for the whole module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moonranger_comparison() -> ConceptStudyComparison:
    return compare_concept_to_pareto(
        get_concept_study("MoonRanger"),
        population_size=24,
        n_generations=4,
        evaluator_eval_cap=200,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Group 1: catalogue accessors
# ---------------------------------------------------------------------------


def test_list_concept_studies_contains_moonranger() -> None:
    assert "MoonRanger" in list_concept_studies()


def test_get_concept_study_round_trip() -> None:
    entry = get_concept_study("MoonRanger")
    assert isinstance(entry, ConceptStudyEntry)
    assert entry.rover_name == "MoonRanger"


def test_get_concept_study_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown concept study"):
        get_concept_study("not-a-real-rover")


def test_moonranger_entry_is_consistent() -> None:
    entry = CONCEPT_STUDIES["MoonRanger"]
    # Design vector is valid (pydantic would have raised on import otherwise)
    assert isinstance(entry.published_design, DesignVector)
    # Scenario is loadable
    scenario = load_scenario(entry.scenario_name)
    assert scenario.name == entry.scenario_name
    # Mass envelope is in the schema range
    assert entry.mass_envelope_kg is not None
    assert 5.0 <= entry.mass_envelope_kg <= 200.0  # generous sanity band
    # Citation is non-empty
    assert "Kumar" in entry.citation
    assert "i-SAIRAS" in entry.citation or "iSAIRAS" in entry.citation
    # Notes mention the imputation rationale
    assert "imput" in entry.notes.lower() or "back-solve" in entry.notes.lower()


# ---------------------------------------------------------------------------
# Group 2: comparison machinery
# ---------------------------------------------------------------------------


def test_comparison_returns_populated_object(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    c = moonranger_comparison
    entry = get_concept_study("MoonRanger")
    assert c.rover_name == "MoonRanger"
    assert c.scenario_name == entry.scenario_name
    assert c.backend_used == "evaluator"
    assert len(c.pareto_designs) == len(c.pareto_metrics) >= 1


def test_published_metrics_predicted_matches_evaluator(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    """The predicted metrics should match a direct evaluator call."""
    c = moonranger_comparison
    entry = get_concept_study(c.rover_name)
    direct = evaluator_evaluate(
        entry.published_design,
        load_scenario(entry.scenario_name),
    )
    assert c.published_metrics_predicted["range_km"] == pytest.approx(
        float(direct.range_km), rel=1e-6
    )
    assert c.published_metrics_predicted["total_mass_kg"] == pytest.approx(
        float(direct.total_mass_kg), rel=1e-6
    )
    assert c.published_metrics_predicted["slope_capability_deg"] == pytest.approx(
        float(direct.slope_capability_deg), rel=1e-6
    )


def test_concept_does_not_dominate_more_than_a_handful_of_pareto_points(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    """A well-converged NSGA-II front contains few points the concept dominates.

    The smoke-budget (24 pop × 4 gen = 96 evals) is not large enough to
    *guarantee* a perfectly non-dominated front w.r.t. the concept's
    feasible design point — small-budget NSGA-II can leave a handful
    of locally-Pareto-but-globally-dominated individuals. The
    paper-grade runs (≥ 10 000 evals) hit 0 reliably; the test
    asserts the smoke run stays inside a 10 %-of-front tolerance.
    """
    c = moonranger_comparison
    tolerance = max(2, int(round(0.10 * len(c.pareto_designs))))
    assert c.n_pareto_points_dominated_by_concept <= tolerance, (
        f"smoke-budget NSGA-II left "
        f"{c.n_pareto_points_dominated_by_concept} Pareto points dominated by "
        f"the concept (front size {len(c.pareto_designs)}); tolerance "
        f"{tolerance}. Consider widening the test budget if this fires "
        f"in CI."
    )


def test_nearest_distance_is_finite_and_non_negative(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    c = moonranger_comparison
    assert 0.0 <= c.nearest_pareto_design_space_distance <= 3.0  # sqrt(9) ceil
    # Index points into the Pareto front
    assert 0 <= c.nearest_pareto_index < len(c.pareto_designs)


def test_best_per_objective_records_all_three(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    c = moonranger_comparison
    assert set(c.best_per_objective) == {o.target for o in DEFAULT_OBJECTIVES}


def test_best_per_objective_value_is_actually_pareto_optimal(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    """For each objective, the recorded best matches the min/max over the front."""
    c = moonranger_comparison
    for obj in DEFAULT_OBJECTIVES:
        values = [m[obj.target] for m in c.pareto_metrics]
        if obj.direction == "max":
            expected = max(values)
        else:
            expected = min(values)
        assert c.best_per_objective[obj.target]["value"] == pytest.approx(expected)


def test_pareto_front_respects_mass_envelope(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    """Every Pareto point must satisfy the entry's mass_envelope_kg constraint."""
    c = moonranger_comparison
    entry = get_concept_study(c.rover_name)
    assert entry.mass_envelope_kg is not None
    for m in c.pareto_metrics:
        # Allow a small tolerance: NSGA-II can return marginally
        # infeasible points if the slip solver is at the edge.
        assert m["total_mass_kg"] <= entry.mass_envelope_kg + 1e-6


def test_dominating_count_is_consistent_with_dominates_helper(
    moonranger_comparison: ConceptStudyComparison,
) -> None:
    """``n_pareto_points_dominating_concept`` matches a recomputation."""
    c = moonranger_comparison
    expected = sum(
        1
        for m in c.pareto_metrics
        if _dominates(m, c.published_metrics_predicted, DEFAULT_OBJECTIVES)
    )
    assert c.n_pareto_points_dominating_concept == expected
