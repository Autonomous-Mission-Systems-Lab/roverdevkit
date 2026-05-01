"""End-to-end mission-evaluator integration tests.

Smoke tests in Week 4: the pipeline runs end-to-end on every canonical
scenario for a Rashid-like design and returns finite, in-range metrics.

The Week 5 acceptance test -- "loaded with Yutu-2 / Pragyan / Rashid
parameters, does the evaluator predict daily traverse distance and
power profile in the right order of magnitude?" -- lives in a separate
notebook per project_plan.md §6 W5.
"""

from __future__ import annotations

import math

import pytest

from roverdevkit.mission.evaluator import evaluate
from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario


@pytest.mark.integration
def test_evaluator_returns_mission_metrics(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    assert isinstance(metrics, MissionMetrics)


@pytest.mark.integration
def test_mass_in_micro_rover_class(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    metrics = evaluate(rashid_like_design, equatorial_scenario)
    # Rashid was ~10 kg; the design vector yields a bottom-up estimate
    # in the 5-50 kg micro-rover class (project_plan.md §1).
    assert 5.0 <= metrics.total_mass_kg <= 50.0


@pytest.mark.integration
def test_all_metrics_are_finite(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    for value in (
        m.range_km,
        m.energy_margin_pct,
        m.slope_capability_deg,
        m.total_mass_kg,
        m.peak_motor_torque_nm,
        m.sinkage_max_m,
    ):
        assert math.isfinite(value)
        assert value >= 0.0


@pytest.mark.integration
def test_range_bounded_by_traverse_distance(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    # range_km cannot exceed traverse_distance_m/1000 -- the sim caps it.
    assert m.range_km <= equatorial_scenario.traverse_distance_m / 1000.0 + 1e-9


@pytest.mark.integration
def test_slope_capability_within_schema_bounds(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    m = evaluate(rashid_like_design, equatorial_scenario)
    # schema allows 0-35 deg
    assert 0.0 <= m.slope_capability_deg <= 35.0


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(list_scenarios()))
def test_evaluator_runs_on_every_scenario(rashid_like_design: DesignVector, name: str) -> None:
    scenario = load_scenario(name)
    metrics = evaluate(rashid_like_design, scenario)
    assert metrics.total_mass_kg > 0.0
    assert metrics.range_km >= 0.0


@pytest.mark.integration
def test_bigger_battery_gives_higher_or_equal_energy_margin(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    baseline = evaluate(rashid_like_design, equatorial_scenario)
    bigger_battery = rashid_like_design.model_copy(update={"battery_capacity_wh": 400.0})
    upgraded = evaluate(bigger_battery, equatorial_scenario)
    # A bigger battery cannot worsen energy margin on the same scenario.
    assert upgraded.energy_margin_pct >= baseline.energy_margin_pct - 1e-6


@pytest.mark.integration
def test_denser_soil_boosts_slope_capability(
    rashid_like_design: DesignVector,
) -> None:
    def make(soil: str, slope: float) -> MissionScenario:
        return MissionScenario(
            name="highland_slope_capability",
            latitude_deg=10.0,
            traverse_distance_m=500.0,
            terrain_class="highland_dense",
            soil_simulant=soil,
            mission_duration_earth_days=5.0,
            max_slope_deg=slope,
        )

    loose = evaluate(rashid_like_design, make("Apollo_regolith_loose", 10.0))
    dense = evaluate(rashid_like_design, make("Apollo_regolith_dense", 10.0))
    assert dense.slope_capability_deg > loose.slope_capability_deg


@pytest.mark.integration
def test_scm_correction_loads_when_artifact_present(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Week-7 step-5: ``use_scm_correction=True`` must successfully run and
    must produce metrics that **differ** from the BW-only baseline when
    the canonical artifact is on disk. If it is absent (e.g. a fresh
    clone before the Week-7 build), the call still succeeds (graceful
    fallback with a warning) but returns BW-identical metrics — which
    is captured here by allowing equality in that case.
    """
    from roverdevkit.terramechanics.correction_model import DEFAULT_CORRECTION_PATH

    bw = evaluate(rashid_like_design, equatorial_scenario, use_scm_correction=False)

    # Either succeeds-and-applies-correction, or warn-and-falls-back to BW.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        scm = evaluate(rashid_like_design, equatorial_scenario, use_scm_correction=True)

    if DEFAULT_CORRECTION_PATH.exists():
        # Production artifact present → at least one mobility-driven
        # metric must move. ``range_km`` saturates at the scenario's
        # traverse-distance ceiling for benign equatorial designs, so
        # pick the unbounded mobility outputs instead.
        moved = (
            scm.sinkage_max_m != bw.sinkage_max_m
            or scm.peak_motor_torque_nm != bw.peak_motor_torque_nm
            or scm.energy_margin_raw_pct != bw.energy_margin_raw_pct
        )
        assert moved, (
            "correction artifact loaded but every mobility-derived metric "
            "matched BW exactly; the wheel-level deltas should perturb at "
            "least one mission-level output"
        )
    else:
        # No artifact on disk → graceful fallback should yield BW-identical metrics.
        assert scm.sinkage_max_m == bw.sinkage_max_m
        assert scm.peak_motor_torque_nm == bw.peak_motor_torque_nm


# ---------------------------------------------------------------------------
# Schema v6/v7 (W12 step B): operational_duty_cycle override
# ---------------------------------------------------------------------------
#
# The pre-v6 ``range_at_utilisation`` post-hoc rescaler is gone. Schema
# v6 plumbed an ``operational_duty_cycle`` override directly into the
# evaluator. Schema v7 collapsed the v6 ``min(δ_des, δ_ops)`` rule
# into ``δ_eff = clamp(δ_ops, [0, 1])`` after ``designed_duty_cycle``
# was removed from the design vector. The tests below pin the v7
# contract.


@pytest.mark.integration
def test_evaluate_default_uses_scenario_operational_duty_cycle(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """``operational_duty_cycle=None`` reproduces the scenario default."""
    metrics_default = evaluate(rashid_like_design, equatorial_scenario)
    metrics_explicit = evaluate(
        rashid_like_design,
        equatorial_scenario,
        operational_duty_cycle=equatorial_scenario.operational_duty_cycle,
    )
    assert math.isclose(metrics_default.range_km, metrics_explicit.range_km, rel_tol=1e-9)


@pytest.mark.integration
def test_lower_operational_duty_cycle_does_not_increase_range(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Halving δ_ops cannot grow forward progress (it scales linearly
    with δ_eff in the kinematic regime; the energy-binding regime
    cancels δ_eff out, in which case range is invariant)."""
    base = evaluate(
        rashid_like_design,
        equatorial_scenario,
        operational_duty_cycle=equatorial_scenario.operational_duty_cycle,
    )
    half = evaluate(
        rashid_like_design,
        equatorial_scenario,
        operational_duty_cycle=0.5 * equatorial_scenario.operational_duty_cycle,
    )
    assert half.range_km <= base.range_km + 1e-9


@pytest.mark.integration
def test_operational_duty_cycle_override_changes_effective_duty(
    rashid_like_design: DesignVector, equatorial_scenario: MissionScenario
) -> None:
    """Schema v7: δ_eff equals the supplied δ_ops (clamped to [0, 1])."""
    from roverdevkit.mission.evaluator import evaluate_verbose

    detailed_low = evaluate_verbose(
        rashid_like_design,
        equatorial_scenario,
        operational_duty_cycle=0.10,
    )
    detailed_high = evaluate_verbose(
        rashid_like_design,
        equatorial_scenario,
        operational_duty_cycle=0.40,
    )
    assert detailed_low.log.effective_duty_cycle == pytest.approx(0.10)
    assert detailed_high.log.effective_duty_cycle == pytest.approx(0.40)
