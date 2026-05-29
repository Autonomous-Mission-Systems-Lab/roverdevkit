"""Drivetrain torque-speed envelope and cruise-speed derivation.

v6 schema update (2026-04-28) overhaul. Background and design rationale:
the version history in ``data/analytical/SCHEMA.md``. Schema v7 (v6 schema update
follow-up) consolidated the v6 designed/operational duty-cycle split
back into a single per-scenario ``operational_duty_cycle``: the v6
``designed_duty_cycle`` field carried no engineering content (the v6
mass model never actually scaled with it) so the only role of
``δ_des`` was to upper-bound ``δ_eff``, which a user can equivalently
express by lowering ``operational_duty_cycle``.

The pre-schema-v7 design vector exposed ``nominal_speed_mps`` and
``drive_duty_cycle`` as free design *inputs* and gated mobility on an
implicit, mass-derived torque ceiling inside the mass model. That made
``range_km`` close to a tautology of two design knobs and let the
optimiser pick rover speeds the drivetrain could not actually sustain
on the scenario soil and slope. This module implements the v6 fix:

1. **Slip-balance** is solved once by the traverse simulator (already
   loop-invariant under the current scenario schema). It returns the
   per-wheel hub torque demand ``T_req`` and equilibrium slip ``s_eq``
   needed to develop the drawbar pull required to climb the scenario's
   worst-case slope plus rolling resistance.
2. **Stall gate** (binary): ``stalled = T_req > peak_wheel_torque_nm``
   or the slip solver failed (no slip in the bracket achieves the
   required DP). Replaces the old implicit "even at slip 0.95 we can't
   develop DP" gate with an explicit, design-controlled torque ceiling.
3. **Energy-balance steady-state cruise speed** ``v_eb``: the speed at
   which avionics + ``δ_eff × P_mobility(v) ≤ P_solar_avg``. Solving
   the equality gives a closed-form ``v_eb`` (no iteration). The
   ``δ_eff`` cancels in the achievable-range product
   ``v_eb × δ_eff × time``, so range in the energy-binding regime is
   independent of duty cycle (only kinematic-bound regimes feel it).
4. **Kinematic envelope** ``v_kin = ω_no_load × R × (1 - s_eq)``. A
   hygiene cap reflecting the conservative drivetrain archetype
   assumption: motor + gearbox combined deliver ``peak_wheel_torque_nm``
   at any hub speed up to ``ω_no_load_hub`` (5 rad/s ≈ 48 rpm). For
   ``R ∈ [0.05, 0.20] m`` and ``s ∈ [0, 0.3]`` this gate is an upper
   bound rarely binding in the energy-binding regime where lunar
   micro-rovers live; we expect to see it fire on < 1 % of LHS samples.
5. **Cruise speed** ``v_cruise = 0`` if stalled, else ``min(v_eb, v_kin)``.

API surface (importable from :mod:`roverdevkit.drivetrain`):

- :data:`OMEGA_NO_LOAD_HUB_RAD_S` — module-level constant for (4).
- :data:`DEFAULT_DRIVETRAIN_EFFICIENCY` — combined motor + gearbox
  efficiency. Mirrors :data:`roverdevkit.mission.traverse_sim.DEFAULT_MOTOR_EFFICIENCY`.
- :func:`effective_duty_cycle` — clamp ``δ_ops`` into ``[0, 1]``
  (kept as a thin helper for symmetry with the v6 API; the previous
  ``min(δ_des, δ_ops)`` semantics collapsed in v7 when
  ``designed_duty_cycle`` was removed from the design vector).
- :func:`kinematic_envelope_v_max` — step (4) closed form.
- :func:`energy_balance_v_cruise` — step (3) closed form.
- :func:`cruise_speed` — composes (2)–(5) into a :class:`CruiseResult`.
- :func:`sizing_peak_torque_anchor_nm` — pre-v6 implicit torque ceiling,
  retained as the LHS prior anchor for the v6 dataset rebuild (so
  ``peak_wheel_torque_nm`` samples cluster around physically plausible
  values for the rest of the design vector).
"""

from __future__ import annotations

from dataclasses import dataclass

OMEGA_NO_LOAD_HUB_RAD_S: float = 5.0
"""No-load hub angular speed of the constant-peak-torque drivetrain
archetype, in rad/s. Approximately 48 rpm at the wheel hub.

The micro-rover regime sits well inside this envelope: at ``R = 0.10 m``
and ``s = 0.10`` the kinematic cap is ``5 × 0.10 × 0.90 = 0.45 m/s``,
roughly an order of magnitude above any lunar-day average cruise. The
constant is exposed at module level (not promoted to a design variable)
because doing so would force the user to think in motor-internal terms
the rest of the design vector deliberately abstracts away. Revisit if
LHS sampling shows > 1 % of cells clamping at ``v_kin_max``."""

DEFAULT_DRIVETRAIN_EFFICIENCY: float = 0.8
"""Combined motor + gearbox efficiency, dimensionless. Mirrors
:data:`roverdevkit.mission.traverse_sim.DEFAULT_MOTOR_EFFICIENCY` so
that the cruise-speed solve and the per-step mobility power use the
same number; keeping a separate copy here would risk silent drift.
Value calibrated against Maxon EC-i + GP series datasheets (BLDC +
planetary gearbox at nominal load)."""


def effective_duty_cycle(operational_duty_cycle: float) -> float:
    """Return ``δ_eff = clamp(δ_ops, [0, 1])``.

    Schema v7 collapsed the v6 ``min(δ_des, δ_ops)`` semantics into a
    single per-scenario duty cycle: the design-side ``designed_duty_cycle``
    field was removed from :class:`~roverdevkit.schema.DesignVector`
    after it turned out to do no engineering work in the mass model.
    This helper is retained as a thin wrapper so callers stay readable
    and so the [0, 1] clamp lives in exactly one place.
    """
    if operational_duty_cycle < 0.0:
        raise ValueError(
            "operational_duty_cycle must be non-negative "
            f"(got {operational_duty_cycle})."
        )
    return min(1.0, operational_duty_cycle)


def kinematic_envelope_v_max(
    omega_no_load_hub_rad_s: float,
    wheel_radius_m: float,
    slip_eq: float,
) -> float:
    """Kinematic cruise-speed cap from the constant-peak-torque envelope.

    ``v_kin = ω_no_load × R × (1 - s_eq)``. The slip term reduces the
    forward speed for a given hub speed: a wheel spinning at ω with
    equilibrium slip ``s`` advances at ``ω × R × (1 - s)``.
    """
    if wheel_radius_m <= 0.0:
        raise ValueError(f"wheel_radius_m must be positive (got {wheel_radius_m}).")
    if omega_no_load_hub_rad_s <= 0.0:
        raise ValueError(
            f"omega_no_load_hub_rad_s must be positive (got {omega_no_load_hub_rad_s})."
        )
    return omega_no_load_hub_rad_s * wheel_radius_m * max(0.0, 1.0 - slip_eq)


def energy_balance_v_cruise(
    *,
    p_solar_avg_w: float,
    p_avionics_w: float,
    wheel_radius_m: float,
    slip_eq: float,
    motor_efficiency: float,
    delta_eff: float,
    n_wheels: int,
    t_req_per_wheel_nm: float,
) -> float:
    """Closed-form energy-balance cruise speed.

    Solves ``δ_eff × P_mobility(v) + P_avionics = P_solar_avg`` for ``v``.
    With ``ω = v / (R × (1 - s_eq))`` and per-wheel mechanical power
    ``T_req × ω``, the per-wheel electrical draw at efficiency η is
    ``T_req × ω / η``; total mobility power is ``n_wheels`` times that.
    Algebraic solve:

        v_eb = (P_solar_avg - P_avionics) × R × (1 - s_eq) × η_motor
               / (δ_eff × n_wheels × T_req)

    Returns 0.0 when net solar headroom is non-positive (rover cannot
    even sustain avionics let alone mobility). Returns ``inf`` when
    ``T_req`` is effectively zero (flat ground, smooth wheels) so that
    callers compose with ``min(v_eb, v_kin_max)`` cleanly.
    """
    if delta_eff < 0.0 or n_wheels <= 0 or wheel_radius_m <= 0.0:
        raise ValueError(
            "delta_eff must be >= 0, n_wheels and wheel_radius_m must be "
            f"positive (got delta_eff={delta_eff}, n_wheels={n_wheels}, "
            f"wheel_radius_m={wheel_radius_m})."
        )
    if motor_efficiency <= 0.0:
        raise ValueError(f"motor_efficiency must be positive (got {motor_efficiency}).")

    p_net_avail = p_solar_avg_w - p_avionics_w
    if p_net_avail <= 0.0:
        return 0.0

    # Effectively-zero torque demand: any speed is energy-feasible, so
    # delegate the binding constraint to the kinematic cap.
    if t_req_per_wheel_nm <= 1e-9:
        return float("inf")

    # delta_eff = 0 means the rover doesn't drive at all; the loop-side
    # multiplier (dx_per_step ∝ δ_eff) zeroes out forward progress
    # regardless of v_eb, but we'd divide by zero here. Return inf so
    # the kinematic cap dominates and the caller gets a finite v_cruise.
    if delta_eff <= 1e-12:
        return float("inf")

    factor = wheel_radius_m * max(1e-6, 1.0 - slip_eq) * motor_efficiency
    return p_net_avail * factor / (delta_eff * n_wheels * t_req_per_wheel_nm)


@dataclass(frozen=True)
class CruiseResult:
    """Output of :func:`cruise_speed`.

    Attributes
    ----------
    stalled
        ``True`` iff the slip solver could not develop the required
        drawbar pull, or the per-wheel torque demand exceeds the
        design's ``peak_wheel_torque_nm``. When ``True``, ``v_cruise_mps``
        is forced to 0.
    v_cruise_mps
        Final cruise speed used by the time loop, m/s.
    v_eb_mps
        Energy-balance solve output, m/s. Stored for diagnostics; can
        be larger than ``v_cruise_mps`` when the kinematic cap binds.
        ``inf`` is possible when ``T_req`` is effectively zero (flat
        ground, smooth wheels).
    v_kin_max_mps
        Kinematic envelope cap, m/s.
    kinematic_clamped
        ``True`` iff ``v_eb`` exceeded ``v_kin_max`` and the cap bound.
        Tracking this lets the LHS dataset builder verify the design
        doc's "< 1 % of cells clamp" assumption.
    delta_eff
        Effective duty cycle the time loop should use.
    """

    stalled: bool
    v_cruise_mps: float
    v_eb_mps: float
    v_kin_max_mps: float
    kinematic_clamped: bool
    delta_eff: float


def cruise_speed(
    *,
    peak_wheel_torque_nm: float,
    t_req_per_wheel_nm: float,
    slip_eq: float,
    slip_solver_failed: bool,
    p_solar_avg_w: float,
    p_avionics_w: float,
    wheel_radius_m: float,
    motor_efficiency: float,
    delta_eff: float,
    n_wheels: int,
    omega_no_load_hub_rad_s: float = OMEGA_NO_LOAD_HUB_RAD_S,
) -> CruiseResult:
    """Compose the stall gate, energy-balance solve, and kinematic cap.

    See module docstring for the physics. This is the canonical entry
    point that :mod:`roverdevkit.mission.traverse_sim` calls on the
    pre-loop wheel-force solve; tests hit it directly.
    """
    if peak_wheel_torque_nm <= 0.0:
        raise ValueError(f"peak_wheel_torque_nm must be positive (got {peak_wheel_torque_nm}).")

    v_kin_max = kinematic_envelope_v_max(
        omega_no_load_hub_rad_s, wheel_radius_m, slip_eq
    )

    stalled = bool(
        slip_solver_failed
        or t_req_per_wheel_nm > peak_wheel_torque_nm + 1e-9
    )
    if stalled:
        return CruiseResult(
            stalled=True,
            v_cruise_mps=0.0,
            v_eb_mps=0.0,
            v_kin_max_mps=v_kin_max,
            kinematic_clamped=False,
            delta_eff=delta_eff,
        )

    v_eb = energy_balance_v_cruise(
        p_solar_avg_w=p_solar_avg_w,
        p_avionics_w=p_avionics_w,
        wheel_radius_m=wheel_radius_m,
        slip_eq=slip_eq,
        motor_efficiency=motor_efficiency,
        delta_eff=delta_eff,
        n_wheels=n_wheels,
        t_req_per_wheel_nm=t_req_per_wheel_nm,
    )

    if v_eb >= v_kin_max:
        return CruiseResult(
            stalled=False,
            v_cruise_mps=v_kin_max,
            v_eb_mps=v_eb,
            v_kin_max_mps=v_kin_max,
            kinematic_clamped=True,
            delta_eff=delta_eff,
        )
    return CruiseResult(
        stalled=False,
        v_cruise_mps=max(0.0, v_eb),
        v_eb_mps=v_eb,
        v_kin_max_mps=v_kin_max,
        kinematic_clamped=False,
        delta_eff=delta_eff,
    )


# ---------------------------------------------------------------------------
# Pre-v6 implicit torque ceiling (LHS prior anchor only)
# ---------------------------------------------------------------------------


def sizing_peak_torque_anchor_nm(
    *,
    total_mass_kg: float,
    wheel_radius_m: float,
    n_wheels: int,
    motor_sizing_safety_factor: float = 2.0,
    motor_peak_friction_coef: float = 0.7,
    gravity_m_per_s2: float = 1.625,
) -> float:
    """Pre-v6 implicit per-wheel torque ceiling, retained as an LHS anchor.

    ``T_anchor = sf × μ × (m × g / N) × R``. In v5 the mass model
    sized motor mass against this ceiling; in v6 ``peak_wheel_torque_nm``
    is a first-class design variable, but the LHS sampler still draws
    around this value (multiplied by a log-uniform tail) so the
    surrogate spends data on physically realisable torque sizings.

    Defaults match
    :class:`roverdevkit.mass.parametric_mers.MassModelParams`. Live
    here (rather than in mass) because the runtime mass model no
    longer computes it; this function exists *only* for the
    LHS prior in :mod:`roverdevkit.surrogate.sampling`.
    """
    if total_mass_kg <= 0.0 or wheel_radius_m <= 0.0 or n_wheels <= 0:
        raise ValueError(
            "total_mass_kg, wheel_radius_m, n_wheels must be positive "
            f"(got total_mass_kg={total_mass_kg}, "
            f"wheel_radius_m={wheel_radius_m}, n_wheels={n_wheels})."
        )
    weight_per_wheel_n = total_mass_kg * gravity_m_per_s2 / n_wheels
    return (
        motor_sizing_safety_factor
        * motor_peak_friction_coef
        * weight_per_wheel_n
        * wheel_radius_m
    )
