"""Tests for the Layer-5 rediscovery harness.

Three groups:

1. **Leakage controls.** Every flown rover maps to one of the
   dedicated class-generic ``*_micro`` scenarios returned by
   :func:`list_class_generic_micro_scenarios`, not to either a
   per-rover validation YAML or one of the canonical tradespace
   scenarios (whose ``operational_duty_cycle`` values were
   inspection-calibrated against real-rover ops history). The
   ``*_micro`` library further enforces a class-neutral δ_ops anchor.
2. **Result-shape contract.** A run on Pragyan (low-budget NSGA-II)
   returns a populated :class:`RediscoveryResult` with the fields the
   downstream report generator depends on.
3. **Determinism.** Two back-to-back runs at the same seed produce
   identical results so the paper figure is reproducible.
"""

from __future__ import annotations

import pytest

from roverdevkit.mission.scenarios import (
    list_class_generic_micro_scenarios,
    list_scenarios,
    load_scenario,
)
from roverdevkit.tradespace.optimizer import DEFAULT_OBJECTIVES
from roverdevkit.validation.rover_registry import flown_registry, registry_by_name
from roverdevkit.validation.rover_rediscovery import (
    RediscoveryResult,
    _CLASS_GENERIC_SCENARIO,
    _MAX_PANEL_TILT_DEG,
    _scenario_panel_orientation,
    class_generic_scenario_for,
    rediscover,
)


# ---------------------------------------------------------------------------
# Leakage controls
# ---------------------------------------------------------------------------


def test_every_flown_rover_has_a_class_generic_scenario() -> None:
    """No flown rover falls back to a canonical tradespace or per-rover YAML."""
    micro_names = set(list_class_generic_micro_scenarios())
    for entry in flown_registry():
        scenario_name = class_generic_scenario_for(entry.rover_name)
        assert scenario_name in micro_names, (
            f"{entry.rover_name} maps to {scenario_name!r}, which is not "
            "one of the class-generic micro-rover scenarios. The "
            "rediscovery test must use the *_micro library only."
        )


def test_class_generic_scenarios_are_never_canonical_or_per_rover() -> None:
    """No mapped scenario name overlaps with the canonical or per-rover scenario sets.

    Both overlap forbidden because:
    - per-rover YAMLs (e.g. ``chandrayaan3_pragyan``) carry rover-
      specific ops calibration. Reusing them leaks the label into the
      search target.
    - canonical tradespace YAMLs (e.g. ``polar_prospecting``) pin
      ``operational_duty_cycle`` to values that were inspection-
      calibrated against real-rover ops history (Pragyan / Yutu-2 /
      Apollo-17 LRV / MER) — a weaker but still real leakage path.
    """
    per_rover_yaml_names = {
        "chandrayaan3_pragyan",
        "change4_yutu2_per_lunar_day",
        "moonranger_polar_demo",
        "rashid_atlas_crater",
        "ispace_m2_tenacious",
        "cadre_polar_unit",
    }
    canonical_names = set(list_scenarios())
    for rover_name, scenario_name in _CLASS_GENERIC_SCENARIO.items():
        assert scenario_name not in per_rover_yaml_names, (
            f"{rover_name} maps to {scenario_name!r}, which is a "
            "per-rover validation YAML. The rediscovery test would "
            "leak per-rover ops calibration into the optimiser."
        )
        assert scenario_name not in canonical_names, (
            f"{rover_name} maps to {scenario_name!r}, which is a "
            "canonical tradespace YAML with an inspection-calibrated "
            "operational_duty_cycle. Use the *_micro library instead."
        )


def test_class_generic_micro_yamls_pin_duty_cycle_to_class_neutral() -> None:
    """Every *_micro scenario uses the class-neutral 0.10 δ_ops anchor.

    This is the rediscovery library's main correctness guarantee: by
    construction no *_micro scenario carries a real-rover ops
    calibration, so δ_ops can never leak into the search target.
    """
    for name in list_class_generic_micro_scenarios():
        scenario = load_scenario(name)
        assert scenario.operational_duty_cycle == pytest.approx(0.10), (
            f"{name} has operational_duty_cycle="
            f"{scenario.operational_duty_cycle}; class-generic micro "
            "scenarios must pin δ_ops to 0.10 (class-neutral)."
        )


def test_class_generic_micro_yamls_are_payload_neutral() -> None:
    """Schema v9: every *_micro scenario ships a zero payload placeholder.

    Scientific payload is a per-rover requirement, so the class-generic
    library must not bake in a payload of its own — the rediscovery
    harness injects each rover's *published* payload onto the scenario
    at run time. A non-zero default here would leak a class-typical
    mass requirement into rovers that carry a different instrument suite.
    """
    for name in list_class_generic_micro_scenarios():
        scenario = load_scenario(name)
        assert scenario.payload_mass_kg == pytest.approx(0.0), (
            f"{name} has payload_mass_kg={scenario.payload_mass_kg}; "
            "class-generic micro scenarios must be payload-neutral."
        )
        assert scenario.payload_power_w == pytest.approx(0.0)


def test_rediscovery_injects_published_payload_mass() -> None:
    """Schema v9: rediscover forwards the rover's published payload mass
    onto the class-generic scenario.

    Verifies the evaluator-level contract the harness relies on: under
    the same class-generic scenario, evaluating the rover's design with
    the published payload yields a modelled total mass higher than the
    payload-free case by exactly the payload (it sits outside the
    dry-mass growth margin).
    """
    from roverdevkit.mission.evaluator import evaluate

    entry = registry_by_name("Pragyan")
    payload = entry.scenario.payload_mass_kg
    assert payload > 0.0, "Pragyan should carry a non-zero instrument payload."

    scenario = load_scenario(class_generic_scenario_for("Pragyan"))
    no_payload = evaluate(entry.design, scenario, payload_mass_kg=0.0)
    with_payload = evaluate(entry.design, scenario, payload_mass_kg=payload)
    assert with_payload.total_mass_kg == pytest.approx(
        no_payload.total_mass_kg + payload, abs=1e-6
    )


def test_class_generic_micro_library_is_complete() -> None:
    """The four *_micro scenarios exist on disk and validate."""
    assert set(list_class_generic_micro_scenarios()) == {
        "polar_micro",
        "mare_micro",
        "highland_micro",
        "crater_rim_micro",
    }


def test_unknown_rover_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="no class-generic scenario"):
        class_generic_scenario_for("Definitely-Not-A-Real-Rover")


# ---------------------------------------------------------------------------
# Result-shape contract (end-to-end smoke against one rover)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pragyan_rediscovery() -> RediscoveryResult:
    """One short-budget rediscovery run cached for the whole module.

    The population size is the binding parameter for small-rover
    rediscovery: random LHS initialisation needs enough candidates to
    contain at least a few mass-feasible designs before NSGA-II's
    feasibility-driven selection can take over. 24 is the smallest
    population that reliably clears Pragyan's ~22 kg modelled budget;
    we also widen ``mass_ceiling_slop`` to 0.20 here purely for test
    headroom (the production default of 0.10 still applies in normal
    usage).
    """
    return rediscover(
        "Pragyan",
        population_size=24,
        n_generations=4,
        mass_ceiling_slop=0.20,
        seed=0,
    )


def test_rediscovery_returns_pareto_front(pragyan_rediscovery: RediscoveryResult) -> None:
    """NSGA-II returns at least one feasible Pareto point."""
    result = pragyan_rediscovery
    assert result.optimization_result.design_vectors
    assert result.optimization_result.metrics
    assert len(result.optimization_result.design_vectors) == len(
        result.optimization_result.metrics
    )


def test_rediscovery_uses_class_generic_scenario(pragyan_rediscovery: RediscoveryResult) -> None:
    result = pragyan_rediscovery
    assert result.class_generic_scenario in list_class_generic_micro_scenarios()
    assert result.class_generic_scenario == "polar_micro"


def test_nearest_pareto_design_is_in_front(pragyan_rediscovery: RediscoveryResult) -> None:
    result = pragyan_rediscovery
    idx = result.nearest_pareto_index
    assert 0 <= idx < len(result.optimization_result.design_vectors)
    assert result.nearest_pareto_design == result.optimization_result.design_vectors[idx]


def test_per_variable_errors_cover_all_continuous_vars(
    pragyan_rediscovery: RediscoveryResult,
) -> None:
    """Every continuous design variable has a signed percent error."""
    expected = {
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
    assert set(result_keys(pragyan_rediscovery.per_variable_percent_errors)) == expected


def test_integer_matches_cover_n_wheels_and_grouser_count(
    pragyan_rediscovery: RediscoveryResult,
) -> None:
    assert set(pragyan_rediscovery.integer_matches) == {"n_wheels", "grouser_count"}


def test_mass_ceiling_constraint_respected(pragyan_rediscovery: RediscoveryResult) -> None:
    """Every Pareto point sits at or below the published mass + 5% ceiling."""
    result = pragyan_rediscovery
    for metric in result.optimization_result.metrics:
        assert metric["total_mass_kg"] <= result.mass_budget_kg + 1e-6


def test_rover_metrics_under_generic_scenario_present(
    pragyan_rediscovery: RediscoveryResult,
) -> None:
    """Reference metrics include every objective target."""
    rover_metrics = pragyan_rediscovery.rover_metrics_under_generic_scenario
    for obj in DEFAULT_OBJECTIVES:
        assert obj.target in rover_metrics


def test_design_space_distance_is_nonnegative(
    pragyan_rediscovery: RediscoveryResult,
) -> None:
    assert pragyan_rediscovery.design_space_distance >= 0.0


# ---------------------------------------------------------------------------
# Determinism (so the paper figure is reproducible)
# ---------------------------------------------------------------------------


def test_rediscovery_is_deterministic_at_fixed_seed() -> None:
    kwargs = {"population_size": 24, "n_generations": 4, "mass_ceiling_slop": 0.20, "seed": 42}
    a = rediscover("Pragyan", **kwargs)
    b = rediscover("Pragyan", **kwargs)
    assert a.nearest_pareto_index == b.nearest_pareto_index
    assert a.design_space_distance == pytest.approx(b.design_space_distance)
    assert a.mass_budget_kg == pytest.approx(b.mass_budget_kg)
    assert a.pareto_dominated == b.pareto_dominated
    for var, va in a.per_variable_percent_errors.items():
        assert va == pytest.approx(b.per_variable_percent_errors[var])


# Tiny helper so the keys-check assertion reads cleanly.
def result_keys(mapping: dict[str, float]) -> set[str]:
    return set(mapping)


# ---------------------------------------------------------------------------
# Panel-orientation fix (2026-05-28): scenario-driven tilt
# ---------------------------------------------------------------------------


def test_scenario_panel_orientation_collapses_to_horizontal_at_equator() -> None:
    """At lat=0 the fixed-tilt approximation is just a horizontal panel."""
    scenario = load_scenario("polar_micro").model_copy(update={"latitude_deg": 0.0})
    tilt, _ = _scenario_panel_orientation(scenario)
    assert tilt == pytest.approx(0.0)


def test_scenario_panel_orientation_caps_polar_tilt() -> None:
    """At lat=±85 the tilt clamps to ``_MAX_PANEL_TILT_DEG`` (80 deg)."""
    south = load_scenario("polar_micro")  # already at lat=-85.0
    north = south.model_copy(update={"latitude_deg": +85.0})
    south_tilt, south_az = _scenario_panel_orientation(south)
    north_tilt, north_az = _scenario_panel_orientation(north)
    assert south_tilt == pytest.approx(_MAX_PANEL_TILT_DEG)
    assert north_tilt == pytest.approx(_MAX_PANEL_TILT_DEG)
    # Southern-hemisphere rovers face local north (azimuth=0); northern
    # rovers face local south (azimuth=180).
    assert south_az == pytest.approx(0.0)
    assert north_az == pytest.approx(180.0)


def test_scenario_panel_orientation_tracks_latitude_below_cap() -> None:
    """Below the cap, tilt = |latitude| so the panel normal points at noon sun."""
    scenario = load_scenario("mare_micro")  # lat=+30
    tilt, az = _scenario_panel_orientation(scenario)
    assert tilt == pytest.approx(30.0)
    assert az == pytest.approx(180.0)


def test_polar_micro_resolves_polar_energy_stall() -> None:
    """Pre-fix the horizontal-panel default sent every polar registry rover
    to ``range_km = 0`` and ``energy_margin_raw_pct < -70 %`` under the
    polar_micro scenario (an ~18x insolation deficit at lat=-85). With
    the scenario-driven panel-tilt fix the rover's own re-evaluation
    now produces *positive* energy margin for the entire polar trio,
    so the rediscovery dominance check is no longer being driven by
    the horizontal-panel modelling artefact.

    Note: range_km > 0 is asserted only for Pragyan and MoonRanger,
    which have enough wheel torque to clear the polar_micro 20-deg
    typical-ops slope. CADRE-unit's 0.06 Nm peak wheel torque
    bottoms out below the scenario's slope-capability threshold
    (12.3 deg < 20 deg), so it still reports range_km=0 and
    stalled=True — but for a *mobility* reason now, not an energy
    one. That's an honest finding rather than a model artefact and
    is documented separately in the comparison report.
    """
    from roverdevkit.validation.rover_rediscovery import _evaluate_rover_under

    polar_scenario = load_scenario("polar_micro")

    # Energy fix: every polar rover now has positive net generation.
    for name in ("Pragyan", "MoonRanger", "CADRE-unit"):
        entry = registry_by_name(name)
        metrics = _evaluate_rover_under(entry, polar_scenario)
        assert metrics["energy_margin_raw_pct"] > 0.0, (
            f"{name}: energy_margin_raw_pct={metrics['energy_margin_raw_pct']:.2f}%, "
            "expected > 0 with the polar panel-tilt fix in place. The "
            "scenario-driven tilt should give the rover ~18x more "
            "insolation than the horizontal-panel default at lat=-85."
        )

    # Mobility check: rovers with enough torque to clear the
    # scenario's 20-deg slope make headway.
    for name in ("Pragyan", "MoonRanger"):
        entry = registry_by_name(name)
        metrics = _evaluate_rover_under(entry, polar_scenario)
        assert metrics["range_km"] > 0.0, (
            f"{name}: range_km={metrics['range_km']:.2f} km, expected > 0 "
            "with positive energy margin and slope-capable hardware "
            "under polar_micro."
        )
