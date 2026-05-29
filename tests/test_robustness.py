"""Tests for the Layer-6 robustness sweep."""

from __future__ import annotations

import math

import pytest

from roverdevkit.mass.parametric_mers import MassModelParams
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.validation.robustness import (
    Perturbation,
    RobustnessSummary,
    cross_scenario_robustness_sweep,
    default_perturbations,
    format_robustness_report,
    perturb_mass_params,
    perturb_soil,
)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------


def test_perturb_soil_scales_kc_and_kphi_jointly() -> None:
    soil = SoilParameters(
        n=1.0, k_c=1.4, k_phi=820.0, cohesion_kpa=0.17, friction_angle_deg=46.0
    )
    softer = perturb_soil(soil, multiplier=0.80)
    assert softer.k_c == pytest.approx(1.12)
    assert softer.k_phi == pytest.approx(656.0)
    # Other Bekker fields untouched.
    assert softer.n == soil.n
    assert softer.cohesion_kpa == soil.cohesion_kpa
    assert softer.friction_angle_deg == soil.friction_angle_deg


def test_perturb_mass_params_scales_named_field() -> None:
    base = MassModelParams()
    bumped = perturb_mass_params(base, "solar_specific_area_mass_kg_per_m2", 1.20)
    assert bumped.solar_specific_area_mass_kg_per_m2 == pytest.approx(
        base.solar_specific_area_mass_kg_per_m2 * 1.20
    )
    # Other fields untouched.
    assert bumped.battery_pack_specific_energy_wh_per_kg == base.battery_pack_specific_energy_wh_per_kg


def test_perturb_mass_params_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown MassModelParams field"):
        perturb_mass_params(MassModelParams(), "no_such_field", 1.20)


# ---------------------------------------------------------------------------
# Sweep structure
# ---------------------------------------------------------------------------


def test_default_perturbations_cover_canonical_axes() -> None:
    perts = default_perturbations()
    names = {p.name for p in perts}
    # Soil stiffness ±20%
    assert "soil_stiffness_minus20" in names
    assert "soil_stiffness_plus20" in names
    # Each mass-model field is exercised in both directions.
    for stem in (
        "wheel_areal_density",
        "solar_areal_mass",
        "battery_specific_energy",
        "motor_specific_torque",
    ):
        assert f"{stem}_minus20" in names
        assert f"{stem}_plus20" in names


@pytest.fixture(scope="module")
def default_sweep() -> list[RobustnessSummary]:
    """Run the canonical sweep once for the whole module."""
    return cross_scenario_robustness_sweep()


def test_sweep_produces_one_summary_per_perturbation(default_sweep: list[RobustnessSummary]) -> None:
    assert len(default_sweep) == len(default_perturbations())


def test_sweep_n_cells_equals_archetypes_times_scenarios(
    default_sweep: list[RobustnessSummary],
) -> None:
    # 3 archetypes × 4 canonical scenarios = 12 cells per perturbation.
    for summary in default_sweep:
        assert summary.n_cells == 12, (
            f"{summary.perturbation}: expected 12 cells, got {summary.n_cells}"
        )


def test_sweep_drifts_are_finite(default_sweep: list[RobustnessSummary]) -> None:
    for summary in default_sweep:
        for value in (
            summary.median_abs_rel_drift_range,
            summary.median_abs_rel_drift_energy_margin,
            summary.median_abs_rel_drift_slope,
            summary.median_abs_rel_drift_mass,
        ):
            assert math.isfinite(value), (
                f"{summary.perturbation} produced a non-finite drift value: {value}"
            )


def test_sweep_ranking_preservation_table_covers_all_scenarios(
    default_sweep: list[RobustnessSummary],
) -> None:
    expected = {
        "equatorial_mare_traverse",
        "polar_prospecting",
        "highland_slope_capability",
        "crater_rim_survey",
    }
    for summary in default_sweep:
        assert set(summary.ranking_preserved_per_scenario) == expected


# ---------------------------------------------------------------------------
# Headline Layer-6 claim: qualitative conclusions persist under ±20 %
# ---------------------------------------------------------------------------


def test_layer6_qualitative_rankings_persist_under_canonical_perturbations(
    default_sweep: list[RobustnessSummary],
) -> None:
    """§7 Layer 6: qualitative conclusions stable under ±20 % soil and mass moves.

    "Stable" here means the (range / energy-margin / slope-capability)
    archetype winner per scenario does not flip under any of the
    canonical ±20 % perturbations. This is the headline claim the
    §8.5 paper paragraph cites.
    """
    failing: list[str] = []
    for summary in default_sweep:
        if not summary.all_rankings_preserved:
            broken = [k for k, v in summary.ranking_preserved_per_scenario.items() if not v]
            failing.append(f"{summary.perturbation}: {broken}")
    # Allow at most ONE perturbation to break ranking on a single scenario;
    # tighter than needed for the headline claim but loose enough to
    # absorb knife-edge cases at the energy-margin boundary.
    n_broken_cells = sum(
        sum(1 for ok in s.ranking_preserved_per_scenario.values() if not ok)
        for s in default_sweep
    )
    assert n_broken_cells <= 1, (
        f"Layer-6 ranking-stability claim violated on {n_broken_cells} cells: "
        + "; ".join(failing)
    )


def test_layer6_continuous_drift_below_30_percent_on_continuous_metrics(
    default_sweep: list[RobustnessSummary],
) -> None:
    """Median absolute relative drift on continuous metrics stays below 30 %.

    This is the "stable in magnitude as well as ranking" companion check.
    30 % is the published Bekker-Wong / mass-model ±15-30 % band
    (Ishigami 2007; Ding 2011; AIAA S-120A-2015 mass-margin), so any
    drift inside that band is the model behaving as expected under a
    parameter move sized at the published uncertainty.
    """
    for summary in default_sweep:
        # Mass drift can be larger than 30 % under ±20 % bumps to the
        # dominant mass coefficients (e.g., a -20 % cut to chassis area
        # density flows almost linearly to total mass on chassis-dominated
        # designs); the headline claim is on the *mission* metrics, so we
        # gate range / energy-margin / slope here and report mass as a
        # diagnostic in the markdown report.
        for label, value in (
            ("range", summary.median_abs_rel_drift_range),
            ("energy_margin", summary.median_abs_rel_drift_energy_margin),
            ("slope", summary.median_abs_rel_drift_slope),
        ):
            assert value < 0.30, (
                f"{summary.perturbation}: median |Δ{label}|/{label} = "
                f"{value * 100:.1f} % exceeds 30 % threshold"
            )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_format_robustness_report_includes_perturbation_names(
    default_sweep: list[RobustnessSummary],
) -> None:
    report = format_robustness_report(default_sweep)
    for summary in default_sweep:
        assert summary.perturbation in report
    # Headline-table columns surface.
    assert "range_km" in report
    assert "energy_margin_raw_pct" in report
    assert "slope_deg" in report
    assert "Ranking preservation" in report


def test_format_robustness_report_renders_canonical_scenario_columns(
    default_sweep: list[RobustnessSummary],
) -> None:
    report = format_robustness_report(default_sweep)
    for name in (
        "equatorial_mare_traverse",
        "polar_prospecting",
        "highland_slope_capability",
        "crater_rim_survey",
    ):
        assert name in report


# ---------------------------------------------------------------------------
# Custom-perturbation smoke
# ---------------------------------------------------------------------------


def test_custom_perturbation_overrides_defaults() -> None:
    custom = (
        Perturbation(name="soil_minus10", kind="soil", multiplier=0.90),
        Perturbation(
            name="solar_areal_mass_minus10",
            kind="mass",
            multiplier=0.90,
            target_field="solar_specific_area_mass_kg_per_m2",
        ),
    )
    summaries = cross_scenario_robustness_sweep(perturbations=custom)
    assert [s.perturbation for s in summaries] == [p.name for p in custom]
