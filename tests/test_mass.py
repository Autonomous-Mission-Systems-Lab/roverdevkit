"""Tests for the bottom-up mass model and its published-rover validation.

Covers:
    - ``MassBreakdown`` construction, totaling, and immutability;
    - ``MassModelParams`` field defaults;
    - per-subsystem physics checks (positivity, monotonicity, linearity);
    - fixed-point iteration convergence (iterations count, and that result
      is independent of the starting guess within tolerance);
    - design-vector wrapper round-tripping through the pydantic schema;
    - the Week-3 validation gate: median absolute percent error on
      in-class rovers must be <= 30 % (plan section 8).
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from roverdevkit.mass import (
    MassBreakdown,
    MassModelParams,
    estimate_mass,
    estimate_mass_from_design,
    validate_against_published_rovers,
)
from roverdevkit.mass.parametric_mers import _wheels_mass
from roverdevkit.schema import DesignVector


def _rashid_like_kwargs(**overrides: Any) -> dict[str, Any]:
    """Reasonable Rashid-class design vector for tests."""
    # Schema v6 (W12 step B): ``peak_wheel_torque_nm`` is a true design
    # input that sizes motor mass directly. 1.0 Nm is in the middle of
    # the Rashid / Pragyan-class hub-torque band.
    base: dict[str, Any] = dict(
        wheel_radius_m=0.10,
        wheel_width_m=0.05,
        n_wheels=4,
        chassis_mass_kg=3.5,
        solar_area_m2=0.4,
        battery_capacity_wh=50.0,
        avionics_power_w=10.0,
        peak_wheel_torque_nm=1.0,
        grouser_height_m=0.005,
        grouser_count=12,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Breakdown / params dataclass behaviour
# ---------------------------------------------------------------------------


class TestMassBreakdown:
    def test_total_equals_sum_of_fields(self) -> None:
        b = MassBreakdown(
            chassis_kg=1.0,
            wheels_kg=1.0,
            motors_and_drives_kg=1.0,
            solar_panels_kg=1.0,
            battery_kg=1.0,
            avionics_kg=1.0,
            harness_kg=1.0,
            thermal_kg=1.0,
            margin_kg=1.0,
        )
        assert b.total_kg == pytest.approx(9.0)
        assert b.dry_kg == pytest.approx(8.0)

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        b = MassBreakdown(
            chassis_kg=1.0,
            wheels_kg=0.0,
            motors_and_drives_kg=0.0,
            solar_panels_kg=0.0,
            battery_kg=0.0,
            avionics_kg=0.0,
            harness_kg=0.0,
            thermal_kg=0.0,
            margin_kg=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            b.chassis_kg = 99.0  # type: ignore[misc]


class TestMassModelParams:
    def test_defaults_are_finite_and_positive(self) -> None:
        p = MassModelParams()
        for field_name in (
            "wheel_structural_area_density_kg_per_m2",
            "grouser_plate_thickness_m",
            "grouser_material_density_kg_per_m3",
            "motor_base_mass_kg",
            "motor_specific_torque_kg_per_nm",
            "motor_peak_friction_coef",
            "motor_sizing_safety_factor",
            "solar_specific_area_mass_kg_per_m2",
            "battery_pack_specific_energy_wh_per_kg",
            "avionics_base_mass_kg",
            "avionics_specific_mass_kg_per_w",
            "harness_fraction",
            "thermal_fraction",
            "margin_fraction",
            "gravity_moon_m_per_s2",
        ):
            value = getattr(p, field_name)
            assert math.isfinite(value) and value > 0, field_name


# ---------------------------------------------------------------------------
# Per-subsystem physics
# ---------------------------------------------------------------------------


class TestWheelsMass:
    def test_positive(self) -> None:
        m = _wheels_mass(
            wheel_radius_m=0.1,
            wheel_width_m=0.05,
            grouser_height_m=0.005,
            grouser_count=12,
            n_wheels=4,
            params=MassModelParams(),
        )
        assert m > 0

    def test_larger_wheel_weighs_more(self) -> None:
        params = MassModelParams()
        small = _wheels_mass(0.08, 0.04, 0.0, 0, 4, params)
        big = _wheels_mass(0.16, 0.08, 0.0, 0, 4, params)
        assert big > small

    def test_more_wheels_weigh_more(self) -> None:
        params = MassModelParams()
        four = _wheels_mass(0.1, 0.05, 0.0, 0, 4, params)
        six = _wheels_mass(0.1, 0.05, 0.0, 0, 6, params)
        assert six == pytest.approx(1.5 * four)

    def test_grouser_mass_linear_in_count(self) -> None:
        params = MassModelParams()
        base = _wheels_mass(0.1, 0.05, 0.005, 0, 4, params)
        m12 = _wheels_mass(0.1, 0.05, 0.005, 12, 4, params)
        m24 = _wheels_mass(0.1, 0.05, 0.005, 24, 4, params)
        assert m12 > base
        assert (m24 - base) == pytest.approx(2 * (m12 - base))

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            dict(wheel_radius_m=0.0, wheel_width_m=0.05),
            dict(wheel_radius_m=-0.1, wheel_width_m=0.05),
            dict(wheel_radius_m=0.1, wheel_width_m=0.0),
            dict(wheel_radius_m=0.1, wheel_width_m=0.05, grouser_height_m=-0.01),
            dict(wheel_radius_m=0.1, wheel_width_m=0.05, grouser_count=-1),
        ],
    )
    def test_rejects_bad_input(self, bad_kwargs: dict[str, Any]) -> None:
        defaults: dict[str, Any] = dict(
            wheel_radius_m=0.1,
            wheel_width_m=0.05,
            grouser_height_m=0.005,
            grouser_count=12,
            n_wheels=4,
        )
        defaults.update(bad_kwargs)
        with pytest.raises(ValueError):
            _wheels_mass(params=MassModelParams(), **defaults)


class TestEstimateMassSubsystemLinearities:
    """``_solar_panels_mass``, ``_battery_mass``, and ``_avionics_mass`` are
    deliberately linear in their respective design variable."""

    def test_solar_mass_linear_in_area(self) -> None:
        b1 = estimate_mass(**_rashid_like_kwargs(solar_area_m2=0.2))
        b2 = estimate_mass(**_rashid_like_kwargs(solar_area_m2=0.4))
        b3 = estimate_mass(**_rashid_like_kwargs(solar_area_m2=0.8))
        assert b2.solar_panels_kg == pytest.approx(2 * b1.solar_panels_kg)
        assert b3.solar_panels_kg == pytest.approx(4 * b1.solar_panels_kg)

    def test_battery_mass_linear_in_capacity(self) -> None:
        b1 = estimate_mass(**_rashid_like_kwargs(battery_capacity_wh=50.0))
        b2 = estimate_mass(**_rashid_like_kwargs(battery_capacity_wh=200.0))
        assert b2.battery_kg == pytest.approx(4 * b1.battery_kg)

    def test_avionics_mass_affine_in_power(self) -> None:
        b1 = estimate_mass(**_rashid_like_kwargs(avionics_power_w=10.0))
        b2 = estimate_mass(**_rashid_like_kwargs(avionics_power_w=30.0))
        # P increases by 20 W -> avionics mass increases by
        # 20 * 0.05 = 1.0 kg per MassModelParams defaults.
        assert b2.avionics_kg - b1.avionics_kg == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Top-level estimate_mass behaviour
# ---------------------------------------------------------------------------


class TestEstimateMass:
    def test_returns_positive_subsystems(self) -> None:
        b = estimate_mass(**_rashid_like_kwargs())
        for attr in (
            "chassis_kg",
            "wheels_kg",
            "motors_and_drives_kg",
            "solar_panels_kg",
            "battery_kg",
            "avionics_kg",
            "harness_kg",
            "thermal_kg",
            "margin_kg",
        ):
            assert getattr(b, attr) > 0, attr
        assert b.total_kg > b.chassis_kg

    def test_iteration_converges_in_a_single_pass(self) -> None:
        # Schema v6 (W12 step B): peak_wheel_torque_nm is a direct design
        # input, so the pre-v6 fixed-point iteration on motor mass vs.
        # total mass is gone and ``n_iterations`` is pinned to 1.
        b = estimate_mass(**_rashid_like_kwargs())
        assert b.n_iterations == 1

    def test_monotonic_in_chassis_mass(self) -> None:
        # Schema v6: motor mass is sized by ``peak_wheel_torque_nm``, not
        # by total vehicle mass, so chassis no longer pulls motor mass
        # along with it. We only assert the direct-additive effect on
        # ``total_kg`` here; motor monotonicity is checked separately.
        b1 = estimate_mass(**_rashid_like_kwargs(chassis_mass_kg=3.0))
        b2 = estimate_mass(**_rashid_like_kwargs(chassis_mass_kg=6.0))
        assert b2.total_kg > b1.total_kg

    def test_motor_mass_monotonic_in_peak_wheel_torque(self) -> None:
        # Schema v6: ``_motors_mass`` is m_0 + k_tau * tau_peak, so
        # bumping the design's peak hub torque must increase motor mass.
        b1 = estimate_mass(**_rashid_like_kwargs(peak_wheel_torque_nm=1.0))
        b2 = estimate_mass(**_rashid_like_kwargs(peak_wheel_torque_nm=4.0))
        assert b2.motors_and_drives_kg > b1.motors_and_drives_kg

    def test_monotonic_in_wheel_size(self) -> None:
        b_small = estimate_mass(**_rashid_like_kwargs(wheel_radius_m=0.08, wheel_width_m=0.04))
        b_big = estimate_mass(**_rashid_like_kwargs(wheel_radius_m=0.18, wheel_width_m=0.10))
        assert b_big.wheels_kg > b_small.wheels_kg

    def test_monotonic_in_battery_capacity(self) -> None:
        b1 = estimate_mass(**_rashid_like_kwargs(battery_capacity_wh=30.0))
        b2 = estimate_mass(**_rashid_like_kwargs(battery_capacity_wh=200.0))
        assert b2.total_kg > b1.total_kg

    def test_rejects_zero_chassis(self) -> None:
        with pytest.raises(ValueError):
            estimate_mass(**_rashid_like_kwargs(chassis_mass_kg=0.0))


class TestEstimateMassFromDesign:
    def test_round_trips_through_design_vector(self, rashid_like_design: DesignVector) -> None:
        b_direct = estimate_mass(
            wheel_radius_m=rashid_like_design.wheel_radius_m,
            wheel_width_m=rashid_like_design.wheel_width_m,
            n_wheels=rashid_like_design.n_wheels,
            chassis_mass_kg=rashid_like_design.chassis_mass_kg,
            solar_area_m2=rashid_like_design.solar_area_m2,
            battery_capacity_wh=rashid_like_design.battery_capacity_wh,
            avionics_power_w=rashid_like_design.avionics_power_w,
            peak_wheel_torque_nm=rashid_like_design.peak_wheel_torque_nm,
            grouser_height_m=rashid_like_design.grouser_height_m,
            grouser_count=rashid_like_design.grouser_count,
        )
        b_via_dv = estimate_mass_from_design(rashid_like_design)
        assert b_via_dv.total_kg == pytest.approx(b_direct.total_kg, rel=1e-9)


# ---------------------------------------------------------------------------
# Published-rover validation gate
# ---------------------------------------------------------------------------


class TestPublishedRoverValidation:
    """The plan's Week-5 validation gate is <= 30 % error on real rovers
    (section 8). We enforce it as a test here at Week-3 on the mass-only
    cross-check to catch regressions early."""

    def test_median_in_class_error_below_30_percent(self) -> None:
        summary = validate_against_published_rovers()
        assert summary.n_in_class >= 4, "need at least 4 in-class rovers to compute median"
        assert summary.median_abs_percent_error_in_class <= 30.0

    def test_no_in_class_rover_worse_than_30_percent(self) -> None:
        summary = validate_against_published_rovers()
        assert abs(summary.worst_in_class.percent_error) <= 30.0

    def test_all_in_class_predictions_positive(self) -> None:
        summary = validate_against_published_rovers()
        for r in summary.per_rover:
            assert r.mass_predicted_kg > 0, r.rover_name

    def test_report_formats(self) -> None:
        from roverdevkit.mass import format_report

        summary = validate_against_published_rovers()
        report = format_report(summary)
        assert "Rashid" in report
        assert "Aggregates" in report
