"""Tests for the shared schema module.

Schema-level validation is the only piece we can fully exercise at project
scaffold time — every other sub-module raises NotImplementedError and will
acquire real tests as it's implemented.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario


def test_design_vector_accepts_valid_inputs(rashid_like_design: DesignVector) -> None:
    assert rashid_like_design.wheel_radius_m == 0.1
    assert rashid_like_design.n_wheels == 4


def test_design_vector_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DesignVector(
            wheel_radius_m=1.0,  # > 0.20 upper bound
            wheel_width_m=0.06,
            grouser_height_m=0.005,
            grouser_count=12,
            n_wheels=4,
            chassis_mass_kg=6.0,
            wheelbase_m=0.35,
            solar_area_m2=0.4,
            battery_capacity_wh=100.0,
            avionics_power_w=15.0,
            peak_wheel_torque_nm=1.5,
        )


def test_design_vector_rejects_invalid_wheel_count() -> None:
    with pytest.raises(ValidationError):
        DesignVector(
            wheel_radius_m=0.1,
            wheel_width_m=0.06,
            grouser_height_m=0.005,
            grouser_count=12,
            n_wheels=5,  # type: ignore[arg-type]  # only 4 or 6 allowed
            chassis_mass_kg=6.0,
            wheelbase_m=0.35,
            solar_area_m2=0.4,
            battery_capacity_wh=100.0,
            avionics_power_w=15.0,
            peak_wheel_torque_nm=1.5,
        )


def test_design_vector_is_immutable(rashid_like_design: DesignVector) -> None:
    with pytest.raises(ValidationError):
        rashid_like_design.wheel_radius_m = 0.2


def test_scenario_round_trips_through_json(equatorial_scenario: MissionScenario) -> None:
    restored = MissionScenario.model_validate_json(equatorial_scenario.model_dump_json())
    assert restored == equatorial_scenario


def test_mission_metrics_constructs() -> None:
    metrics = MissionMetrics(
        range_km=3.2,
        energy_margin_pct=18.5,
        slope_capability_deg=17.0,
        total_mass_kg=10.8,
        peak_motor_torque_nm=1.4,
        sinkage_max_m=0.012,
        thermal_survival=True,
        stalled=False,
    )
    assert metrics.range_km_std is None
