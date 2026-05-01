"""Drivetrain modelling.

This package owns the motor + gearbox torque-speed envelope and the
helpers that derive cruise speed inside the mission evaluator. See
:mod:`roverdevkit.drivetrain.motor` for the public API.
"""

from roverdevkit.drivetrain.motor import (
    DEFAULT_DRIVETRAIN_EFFICIENCY,
    OMEGA_NO_LOAD_HUB_RAD_S,
    CruiseResult,
    cruise_speed,
    effective_duty_cycle,
    energy_balance_v_cruise,
    kinematic_envelope_v_max,
    sizing_peak_torque_anchor_nm,
)

__all__ = [
    "DEFAULT_DRIVETRAIN_EFFICIENCY",
    "OMEGA_NO_LOAD_HUB_RAD_S",
    "CruiseResult",
    "cruise_speed",
    "effective_duty_cycle",
    "energy_balance_v_cruise",
    "kinematic_envelope_v_max",
    "sizing_peak_torque_anchor_nm",
]
