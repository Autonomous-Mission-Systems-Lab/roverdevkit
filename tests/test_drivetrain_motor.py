"""Unit tests for :mod:`roverdevkit.drivetrain.motor`.

The drivetrain module is the v6 W12-step-B replacement for the
implicit torque ceiling that used to live inside the mass model and
the implicit cruise speed that used to live on the design vector.
These tests exercise the four pieces independently:

* :func:`effective_duty_cycle` (clamp, bad inputs)
* :func:`kinematic_envelope_v_max` (closed-form, slip term)
* :func:`energy_balance_v_cruise` (closed-form, edge cases)
* :func:`cruise_speed` (composer; stall gate, kinematic clamp,
  energy-binding regime)
* :func:`sizing_peak_torque_anchor_nm` (LHS prior)
"""

from __future__ import annotations

import math

import pytest

from roverdevkit.drivetrain.motor import (
    DEFAULT_DRIVETRAIN_EFFICIENCY,
    OMEGA_NO_LOAD_HUB_RAD_S,
    cruise_speed,
    effective_duty_cycle,
    energy_balance_v_cruise,
    kinematic_envelope_v_max,
    sizing_peak_torque_anchor_nm,
)


# ---------------------------------------------------------------------------
# effective_duty_cycle
# ---------------------------------------------------------------------------


def test_effective_duty_cycle_passes_through() -> None:
    # Schema v7 collapsed the v6 ``min(δ_des, δ_ops)`` semantics into
    # a thin clamp on a single per-scenario ``operational_duty_cycle``.
    assert effective_duty_cycle(0.4) == pytest.approx(0.4)
    assert effective_duty_cycle(0.3) == pytest.approx(0.3)


def test_effective_duty_cycle_clamps_above_one() -> None:
    # Caller should never pass >1, but the clamp is a belt-and-braces
    # guard against wonky overrides.
    assert effective_duty_cycle(1.5) == pytest.approx(1.0)


def test_effective_duty_cycle_zero_is_zero() -> None:
    assert effective_duty_cycle(0.0) == 0.0


def test_effective_duty_cycle_rejects_negative() -> None:
    with pytest.raises(ValueError):
        effective_duty_cycle(-0.1)
    with pytest.raises(ValueError):
        effective_duty_cycle(-0.01)


# ---------------------------------------------------------------------------
# kinematic_envelope_v_max
# ---------------------------------------------------------------------------


def test_kinematic_envelope_zero_slip_is_omega_R() -> None:
    v = kinematic_envelope_v_max(5.0, 0.10, 0.0)
    assert v == pytest.approx(0.5)


def test_kinematic_envelope_slip_reduces_speed_linearly() -> None:
    # v(s) / v(0) = 1 - s
    v0 = kinematic_envelope_v_max(5.0, 0.10, 0.0)
    v = kinematic_envelope_v_max(5.0, 0.10, 0.30)
    assert v / v0 == pytest.approx(0.70)


def test_kinematic_envelope_clamps_at_zero_for_extreme_slip() -> None:
    # Slip > 1 is non-physical but should not produce negative speeds.
    v = kinematic_envelope_v_max(5.0, 0.10, 1.5)
    assert v == 0.0


def test_kinematic_envelope_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        kinematic_envelope_v_max(5.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        kinematic_envelope_v_max(0.0, 0.10, 0.1)


# ---------------------------------------------------------------------------
# energy_balance_v_cruise
# ---------------------------------------------------------------------------


def test_energy_balance_closed_form_matches_hand_solve() -> None:
    # Hand solve: P_net = 50 W, R = 0.10 m, s = 0.10, η = 0.8,
    # δ_eff = 0.5, N = 4, T = 2 Nm
    # v = 50 * 0.10 * 0.90 * 0.8 / (0.5 * 4 * 2)
    #   = 3.6 / 4.0 = 0.90 m/s
    v = energy_balance_v_cruise(
        p_solar_avg_w=60.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.10,
        motor_efficiency=0.8,
        delta_eff=0.5,
        n_wheels=4,
        t_req_per_wheel_nm=2.0,
    )
    assert v == pytest.approx(0.90, rel=1e-9)


def test_energy_balance_zero_when_no_solar_headroom() -> None:
    v = energy_balance_v_cruise(
        p_solar_avg_w=5.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.10,
        motor_efficiency=0.8,
        delta_eff=0.5,
        n_wheels=4,
        t_req_per_wheel_nm=2.0,
    )
    assert v == 0.0


def test_energy_balance_inf_when_torque_demand_zero() -> None:
    # Flat ground, smooth wheels — kinematic cap should bind, not
    # energy balance. Returning inf lets the composer use min().
    v = energy_balance_v_cruise(
        p_solar_avg_w=60.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.0,
        motor_efficiency=0.8,
        delta_eff=0.5,
        n_wheels=4,
        t_req_per_wheel_nm=0.0,
    )
    assert math.isinf(v)


def test_energy_balance_inf_when_delta_eff_zero() -> None:
    # No driving duty -> dx_per_step is zero on the loop side; the
    # cruise speed itself just feeds the kinematic cap.
    v = energy_balance_v_cruise(
        p_solar_avg_w=60.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.10,
        motor_efficiency=0.8,
        delta_eff=0.0,
        n_wheels=4,
        t_req_per_wheel_nm=2.0,
    )
    assert math.isinf(v)


def test_energy_balance_scales_inversely_with_delta_eff() -> None:
    # Doubling δ_eff halves v_eb (energy budget for mobility halves).
    common = dict(
        p_solar_avg_w=60.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.10,
        motor_efficiency=0.8,
        n_wheels=4,
        t_req_per_wheel_nm=2.0,
    )
    v1 = energy_balance_v_cruise(delta_eff=0.25, **common)
    v2 = energy_balance_v_cruise(delta_eff=0.50, **common)
    assert v1 / v2 == pytest.approx(2.0, rel=1e-9)


def test_energy_balance_rejects_bad_inputs() -> None:
    common = dict(
        p_solar_avg_w=60.0,
        p_avionics_w=10.0,
        wheel_radius_m=0.10,
        slip_eq=0.10,
        delta_eff=0.5,
        n_wheels=4,
        t_req_per_wheel_nm=2.0,
    )
    with pytest.raises(ValueError):
        energy_balance_v_cruise(motor_efficiency=0.0, **common)
    with pytest.raises(ValueError):
        energy_balance_v_cruise(motor_efficiency=0.8, **{**common, "wheel_radius_m": 0.0})
    with pytest.raises(ValueError):
        energy_balance_v_cruise(motor_efficiency=0.8, **{**common, "n_wheels": 0})


# ---------------------------------------------------------------------------
# cruise_speed (composer)
# ---------------------------------------------------------------------------


_BASE_KWARGS = dict(
    peak_wheel_torque_nm=5.0,
    t_req_per_wheel_nm=2.0,
    slip_eq=0.10,
    slip_solver_failed=False,
    p_solar_avg_w=60.0,
    p_avionics_w=10.0,
    wheel_radius_m=0.10,
    motor_efficiency=0.8,
    delta_eff=0.5,
    n_wheels=4,
)


def test_cruise_speed_energy_binding_regime() -> None:
    # v_kin = 5 * 0.10 * 0.90 = 0.45 m/s. Lower solar headroom so
    # v_eb < v_kin and the energy-binding branch is exercised.
    # P_net = 15 - 10 = 5 W → v_eb = 5 * 0.10 * 0.90 * 0.8 / (0.5*4*2)
    #   = 0.36 / 4.0 = 0.09 m/s.
    res = cruise_speed(**{**_BASE_KWARGS, "p_solar_avg_w": 15.0})
    assert not res.stalled
    assert not res.kinematic_clamped
    assert res.v_cruise_mps == pytest.approx(0.09, rel=1e-9)
    assert res.v_eb_mps == pytest.approx(0.09, rel=1e-9)
    assert res.v_kin_max_mps == pytest.approx(0.45, rel=1e-9)


def test_cruise_speed_kinematic_clamp_fires() -> None:
    res = cruise_speed(**_BASE_KWARGS)
    # v_kin = 5 * 0.10 * 0.90 = 0.45 < v_eb = 0.90 -> clamped.
    assert not res.stalled
    assert res.kinematic_clamped
    assert res.v_cruise_mps == pytest.approx(0.45, rel=1e-9)
    assert res.v_kin_max_mps == pytest.approx(0.45, rel=1e-9)
    assert res.v_eb_mps > res.v_kin_max_mps


def test_cruise_speed_stall_gate_torque_excess() -> None:
    res = cruise_speed(**{**_BASE_KWARGS, "peak_wheel_torque_nm": 1.0})
    assert res.stalled
    assert res.v_cruise_mps == 0.0


def test_cruise_speed_stall_gate_solver_failed() -> None:
    res = cruise_speed(**{**_BASE_KWARGS, "slip_solver_failed": True})
    assert res.stalled
    assert res.v_cruise_mps == 0.0


def test_cruise_speed_stalled_when_no_solar_headroom() -> None:
    # Avionics > solar avg → v_eb collapses to 0; rover isn't formally
    # "stalled" by torque, but cruise speed is 0. Verify that.
    res = cruise_speed(**{**_BASE_KWARGS, "p_solar_avg_w": 5.0})
    assert not res.stalled
    assert res.v_cruise_mps == 0.0


def test_cruise_speed_zero_torque_demand_uses_kinematic_cap() -> None:
    # Flat ground / smooth tires → energy balance is unbounded; the
    # kinematic envelope is the only finite cap.
    res = cruise_speed(**{**_BASE_KWARGS, "t_req_per_wheel_nm": 0.0})
    assert not res.stalled
    assert res.kinematic_clamped
    assert res.v_cruise_mps == pytest.approx(0.45, rel=1e-9)


def test_cruise_speed_omega_constant_default_value() -> None:
    # Locks in the documented constant; if anyone bumps it the design
    # doc and the about-dialog need to follow.
    assert OMEGA_NO_LOAD_HUB_RAD_S == 5.0


def test_default_drivetrain_efficiency_matches_traverse_sim() -> None:
    # Two copies of the constant must stay in sync; surrogate quality
    # gates depend on the cruise solve and the per-step mobility power
    # using the same η.
    from roverdevkit.mission.traverse_sim import DEFAULT_MOTOR_EFFICIENCY

    assert DEFAULT_DRIVETRAIN_EFFICIENCY == DEFAULT_MOTOR_EFFICIENCY


def test_cruise_speed_rejects_bad_peak_torque() -> None:
    with pytest.raises(ValueError):
        cruise_speed(**{**_BASE_KWARGS, "peak_wheel_torque_nm": 0.0})


# ---------------------------------------------------------------------------
# sizing_peak_torque_anchor_nm
# ---------------------------------------------------------------------------


def test_sizing_peak_torque_anchor_matches_v5_formula() -> None:
    # T = sf * mu * (m * g / N) * R = 2.0 * 0.7 * (15 * 1.625 / 4) * 0.10
    #   = 2.0 * 0.7 * 6.09375 * 0.10 = 0.853125
    t = sizing_peak_torque_anchor_nm(
        total_mass_kg=15.0,
        wheel_radius_m=0.10,
        n_wheels=4,
    )
    assert t == pytest.approx(0.853125, rel=1e-9)


def test_sizing_peak_torque_anchor_scales_with_mass() -> None:
    t1 = sizing_peak_torque_anchor_nm(total_mass_kg=10.0, wheel_radius_m=0.10, n_wheels=4)
    t2 = sizing_peak_torque_anchor_nm(total_mass_kg=20.0, wheel_radius_m=0.10, n_wheels=4)
    assert t2 / t1 == pytest.approx(2.0, rel=1e-9)


def test_sizing_peak_torque_anchor_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        sizing_peak_torque_anchor_nm(total_mass_kg=0.0, wheel_radius_m=0.1, n_wheels=4)
    with pytest.raises(ValueError):
        sizing_peak_torque_anchor_nm(total_mass_kg=10.0, wheel_radius_m=0.0, n_wheels=4)
    with pytest.raises(ValueError):
        sizing_peak_torque_anchor_nm(total_mass_kg=10.0, wheel_radius_m=0.1, n_wheels=0)
