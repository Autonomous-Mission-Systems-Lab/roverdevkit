"""Cross-scenario ranking + one-at-a-time sensitivity tests.

Tests the "is the evaluator right in the right *direction* when
parameters change?" requirement from project_plan.md §6 W5.

Split into two families:
1. Ranking invariants - qualitative comparisons between hand-crafted
   archetype designs on canonical scenarios.
2. Sensitivity invariants - signs of metric deltas under single-variable
   perturbations.

These are intentionally direction-only tests. Magnitudes would couple
to specific parameter values and become brittle to calibration.

Performance
-----------
The two helpers (``rank_archetypes`` and ``one_at_a_time_sensitivity``)
each run the evaluator dozens of times. The session-scoped fixtures in
``conftest.py`` (``cross_scenario_rankings``,
``cross_scenario_sensitivity_by_var``) cache the outputs so all tests
in this module share a single evaluator pass. This drops module
runtime from ~280 s to ~10 s.
"""

from __future__ import annotations

from roverdevkit.validation.cross_scenario import ArchetypeRanking, SensitivityEntry

# ---------------------------------------------------------------------------
# Ranking invariants
# ---------------------------------------------------------------------------


def test_slope_climber_has_highest_slope_capability_everywhere(
    cross_scenario_rankings: dict[str, ArchetypeRanking],
) -> None:
    """On every canonical scenario, the slope-specialty archetype must
    out-climb the other two. This validates that the slope-capability
    calculation is sensitive to the expected design levers (big wheels,
    many wheels, deep grousers)."""
    for scen, rank in cross_scenario_rankings.items():
        slope_cap = {name: m.slope_capability_deg for name, m in rank.per_archetype.items()}
        assert slope_cap["slope_climber"] >= slope_cap["large_traverser"], scen
        assert slope_cap["slope_climber"] >= slope_cap["polar_survivor"], scen
        assert rank.slope_capability_winner == "slope_climber", scen


def test_large_traverser_has_highest_or_tied_range_in_benign_scenarios(
    cross_scenario_rankings: dict[str, ArchetypeRanking],
) -> None:
    """On scenarios where range is not catastrophically capped, the
    fast/big/duty-heavy archetype should tie or beat the slow archetype."""
    eq_rank = cross_scenario_rankings["equatorial_mare_traverse"]
    range_km = {name: m.range_km for name, m in eq_rank.per_archetype.items()}
    assert range_km["large_traverser"] >= range_km["polar_survivor"]


def test_every_canonical_scenario_evaluates_all_archetypes(
    cross_scenario_rankings: dict[str, ArchetypeRanking],
) -> None:
    """Guard against silent evaluator crashes - must get 3 metrics per scenario."""
    assert len(cross_scenario_rankings) == 4
    for rank in cross_scenario_rankings.values():
        assert set(rank.per_archetype.keys()) == {
            "large_traverser",
            "polar_survivor",
            "slope_climber",
        }


def test_total_mass_is_consistent_across_scenarios(
    cross_scenario_rankings: dict[str, ArchetypeRanking],
) -> None:
    """Total mass depends only on the design vector, not the scenario.

    A surprising cross-scenario mass variance would signal that the
    evaluator is coupling mass to mission duration or terrain - a bug.
    """
    masses_per_archetype: dict[str, set[float]] = {}
    for rank in cross_scenario_rankings.values():
        for name, m in rank.per_archetype.items():
            masses_per_archetype.setdefault(name, set()).add(round(m.total_mass_kg, 6))
    for name, masses in masses_per_archetype.items():
        assert len(masses) == 1, f"{name}: total_mass_kg varies across scenarios {masses}"


# ---------------------------------------------------------------------------
# Sensitivity invariants (one-at-a-time)
# ---------------------------------------------------------------------------


def test_more_avionics_power_lowers_energy_margin(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """avionics_power_w is a pure parasitic load; margin must drop."""
    entry = cross_scenario_sensitivity_by_var["avionics_power_w"]
    assert entry.delta_energy_margin_pct < 0.0


def test_more_chassis_mass_increases_total_mass(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """chassis_mass_kg is part of total_mass_kg directly plus propagates
    through the mass-model fixed point."""
    entry = cross_scenario_sensitivity_by_var["chassis_mass_kg"]
    assert entry.delta_total_mass_kg > 0.0


def test_bigger_wheel_radius_raises_slope_capability(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """Bigger wheels -> more contact area -> more DP on a given soil.

    The wheel_radius lever is the single largest slope-capability
    modifier in the design space."""
    entry = cross_scenario_sensitivity_by_var["wheel_radius_m"]
    assert entry.delta_slope_capability_deg > 0.5


def test_more_peak_wheel_torque_increases_total_mass(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """Schema v6: peak_wheel_torque_nm sizes the motor and gearbox mass
    contributions in the parametric mass model. Bumping torque capacity
    must therefore increase total dry mass.
    """
    entry = cross_scenario_sensitivity_by_var["peak_wheel_torque_nm"]
    assert entry.delta_total_mass_kg > 0.0


def test_bigger_solar_area_never_decreases_energy_margin(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """More solar in -> margin same or higher. Small bumps may saturate
    a high-margin baseline; the monotonicity direction must hold."""
    entry = cross_scenario_sensitivity_by_var["solar_area_m2"]
    assert entry.delta_energy_margin_pct >= -1e-6


def test_bigger_battery_never_decreases_energy_margin(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """More stored energy can only help energy-margin, all else equal."""
    entry = cross_scenario_sensitivity_by_var["battery_capacity_wh"]
    assert entry.delta_energy_margin_pct >= -1e-6


# ---------------------------------------------------------------------------
# Raw (unclipped) energy-margin invariants
# ---------------------------------------------------------------------------
# The clipped SOC-based margin saturates at 0 % and 100 %; the raw
# integrated-energy margin must remain sensitive to generation /
# consumption levers even in saturated regimes. These tests guard the
# Phase-2 surrogate-training signal added in schema.MissionMetrics.


def test_raw_energy_margin_strictly_increases_with_solar_area(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """Bigger solar area on a saturated-margin baseline: clipped metric
    may read +0.00 but the integrated-energy margin must be strictly
    positive, because more generation against the same consumption can
    only grow the surplus."""
    entry = cross_scenario_sensitivity_by_var["solar_area_m2"]
    assert entry.delta_energy_margin_raw_pct > 1e-6


def test_raw_energy_margin_strictly_decreases_with_avionics_power(
    cross_scenario_sensitivity_by_var: dict[str, SensitivityEntry],
) -> None:
    """Parasitic load must drop the integrated-energy margin; this is
    the main Phase-2 signal the surrogate will learn."""
    entry = cross_scenario_sensitivity_by_var["avionics_power_w"]
    assert entry.delta_energy_margin_raw_pct < -1e-6
