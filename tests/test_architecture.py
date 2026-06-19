"""Tests for the mobility-architecture proxy."""

from __future__ import annotations

import pytest

from roverdevkit.architecture import (
    architecture_suspension_mass_kg,
    obstacle_capability_m,
    obstacle_margin_m,
    obstacle_requirement_met,
    wheel_count_for_architecture,
)
from roverdevkit.mass.parametric_mers import estimate_mass_from_design
from roverdevkit.schema import DesignVector


def test_wheel_count_for_architecture() -> None:
    assert wheel_count_for_architecture("rigid_4wheel") == 4
    assert wheel_count_for_architecture("rocker_bogie_6wheel") == 6


def test_obstacle_capability_scales_with_architecture() -> None:
    rigid = obstacle_capability_m("rigid_4wheel", 0.10)
    rocker = obstacle_capability_m("rocker_bogie_6wheel", 0.10)
    assert rocker > rigid
    assert rigid == pytest.approx(0.05)
    assert rocker == pytest.approx(0.125)


def test_rocker_bogie_adds_suspension_mass() -> None:
    design = DesignVector(
        mobility_architecture="rocker_bogie_6wheel",
        wheel_radius_m=0.1,
        wheel_width_m=0.06,
        grouser_height_m=0.005,
        grouser_count=12,
        n_wheels=6,
        chassis_mass_kg=10.0,
        wheelbase_m=0.5,
        solar_area_m2=0.4,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=1.5,
    )
    rigid = design.model_copy(
        update={"mobility_architecture": "rigid_4wheel", "n_wheels": 4}
    )
    m_rocker = estimate_mass_from_design(design).total_kg
    m_rigid = estimate_mass_from_design(rigid).total_kg
    assert m_rocker > m_rigid
    assert architecture_suspension_mass_kg("rocker_bogie_6wheel", 10.0) > 0.0


def test_obstacle_margin_and_requirement() -> None:
    cap = obstacle_capability_m("rigid_4wheel", 0.20)
    assert obstacle_margin_m(cap, 0.05) == pytest.approx(cap - 0.05)
    assert obstacle_requirement_met(cap, 0.05)
    assert not obstacle_requirement_met(cap, cap + 0.01)
