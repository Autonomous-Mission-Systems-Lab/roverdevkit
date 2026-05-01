"""Time-stepped traverse simulator.

Given a rover design, a mission scenario, soil parameters, and the total
vehicle mass, this module marches the rover forward in fixed time steps:
at each step it solves the Bekker-Wong slip balance on the scenario's
slope, draws mobility power from the battery, replenishes from the
solar panel, and logs everything.

The simulator **always runs to the end of the mission duration**; it
does not short-circuit when the battery hits its DoD floor or the
rover stalls. Early termination would throw away information the
surrogate layer (project_plan.md §4) needs to learn failure modes. The
end-of-run constraint flags and ``terminated_reason`` field capture
whatever failures occurred during the run.

Integration notes
-----------------
- At each step we solve ``DP(slip) - DP_required_per_wheel = 0`` via
  :func:`scipy.optimize.brentq` bracketed in ``[-0.9, 0.95]``. If no
  root exists (the slope is unclimbable), slip is pinned at the upper
  bracket and effective forward velocity drops to zero -- the rover
  spins in place, still drawing motor power.
- We apply the *effective* duty cycle ``δ_eff = min(designed,
  operational)`` as a mission-average scaling on mobility power and
  forward progress. This is the standard tradespace approximation
  (project_plan.md §4); pinning down a drive schedule is deferred to
  v2. **W12 Step A (2026-04-27) addendum:** when the battery hits its
  DoD floor and the unthrottled power balance goes negative, the
  per-step mobility duty is locally throttled to whatever fraction of
  ``δ_eff`` the instantaneous solar input can sustain
  (``min(δ_eff, (p_solar - p_avionics) / p_drive)``). This makes
  ``range_km`` an *energy-feasible* metric rather than a capability
  envelope; details in ``reports/week12_design/decision.md``.
- **W12 Step B (2026-04-28) addendum:** cruise speed is now *derived*
  inside :func:`run_traverse` from the slip-balance torque demand,
  the mission-average solar power budget, the kinematic envelope, and
  the design's ``peak_wheel_torque_nm`` — see
  :mod:`roverdevkit.drivetrain.motor`. ``DesignVector.nominal_speed_mps``
  is gone. Schema v7 (W12 step B follow-up) further removed
  ``designed_duty_cycle`` from the design vector after that field
  turned out to do no engineering work in the v6 mass model; the
  per-scenario / override ``operational_duty_cycle`` is now used
  directly as ``δ_eff`` (clamped to ``[0, 1]``).
- Thermal survival is treated as a whole-mission binary flag
  (:mod:`roverdevkit.power.thermal`) rather than a per-step check --
  the lumped-parameter model is steady-state.

Performance (post-W7.7 lift-out, project_log.md 2026-04-26 entry)
-----------
Default ``dt_s = 3600`` (1 hour) gives ~340 steps for a 14-day mission
and ~720 steps for a 30-day mission. The Bekker-Wong slip solve and the
mobility-power calculation are **loop-invariant** under the current
flat-slope / fixed-soil scenario schema (their inputs do not change
across mission steps), so they are computed once *before* the time
loop and reused. Per-step cost in the time loop is therefore dominated
by the cheap solar / battery / kinematic update.

End-to-end mission cost on Apple Silicon (single core):

* analytical path (BW only):                   ~30 ms / mission
* analytical + wheel-level SCM correction:     ~40 ms / mission
* SCM-direct (PyChrono in the wheel-force solve, available via the
  ``force_backend="scm"`` switch for validation only):  ~4 s / mission

The 100× gap between BW + correction and SCM-direct, with W7.7 showing
≥99 % feasibility-flip agreement, is the cost-vs-fidelity result that
makes the wheel-level correction architecture the production choice.

Future scenario schemas that vary slope or soil per mission step will
need to move the lifted-out calls back inside the loop; the
``force_backend`` / ``correction`` plumbing is already structured for
that.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from roverdevkit.drivetrain.motor import (
    CruiseResult,
    cruise_speed,
    effective_duty_cycle,
)
from roverdevkit.power.battery import BatteryState
from roverdevkit.power.battery import step as battery_step
from roverdevkit.power.solar import (
    LUNAR_SYNODIC_DAY_HOURS,
    lunar_hour_angle_deg,
    panel_power_w,
    sun_azimuth_deg,
    sun_elevation_deg,
)
from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelForces,
    WheelGeometry,
    single_wheel_forces,
)
from roverdevkit.terramechanics.correction_model import (
    FEATURE_COLUMNS as _CORR_FEATURE_COLUMNS,
)
from roverdevkit.terramechanics.correction_model import (
    WheelLevelCorrection,
)

# Index of the slip feature in the wheel-level correction's feature
# vector. Cached so the per-step brentq residual can mutate just one
# entry of a pre-built feature buffer instead of rebuilding the whole
# vector each call. If the schema ever changes this assert will fail
# loudly at import time rather than silently corrupting predictions.
_CORR_SLIP_IDX: int = _CORR_FEATURE_COLUMNS.index("slip")
assert _CORR_FEATURE_COLUMNS == (
    "vertical_load_n",
    "slip",
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    "soil_n",
    "soil_k_c",
    "soil_k_phi",
    "soil_cohesion_kpa",
    "soil_friction_angle_deg",
    "soil_shear_modulus_k_m",
), "correction-model feature schema changed; update _build_feature_buffer in traverse_sim.py"

DEFAULT_MOTOR_EFFICIENCY: float = 0.8
"""Electrical-to-mechanical drivetrain efficiency (motor + gearbox).

0.8 is mid-range for a space-qualified brushless motor + planetary
gearbox pair at nominal load (Maxon EC-i + GP series datasheets)."""

DEFAULT_PANEL_EFFICIENCY: float = 0.28
"""Default DC conversion efficiency of a GaAs triple-junction panel.

Matches the upper end of flight-heritage cells (Spectrolab XTJ, ZTJ).
Override via ``panel_efficiency`` if the design specifies a different
cell technology."""

DEFAULT_PANEL_DUST_FACTOR: float = 0.90
"""Dust-degradation factor; 10 % loss is a reasonable tradespace default
for a few lunar days of operation (Yutu-2 showed ~10-15 %)."""

DEFAULT_DT_S: float = 3600.0
"""Default time step, s. One Earth hour."""

_SLIP_LOWER_BOUND: float = -0.9
_SLIP_UPPER_BOUND: float = 0.95
"""Brentq search bracket for the per-step slip solver. Reflects the
physical limits at which the Bekker-Wong model is credible."""


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


@dataclass
class TraverseLog:
    """Per-step traverse-sim history arrays plus termination metadata.

    Schema v6 (W12 step B): three new top-level fields make the
    derived-cruise-speed pipeline observable to callers:

    - ``cruise_speed_mps`` — the rover speed the time loop actually
      drove at, returned by
      :func:`roverdevkit.drivetrain.motor.cruise_speed`.
    - ``effective_duty_cycle`` — schema v7: ``operational_duty_cycle``
      (per-scenario default, or per-call override) clamped to
      ``[0, 1]``. The v6 ``min(δ_des, δ_ops)`` semantics collapsed
      when ``designed_duty_cycle`` was removed from the design vector.
    - ``cruise_kinematic_clamped`` — ``True`` when the kinematic
      envelope cap (not the energy-balance solve) bound; tracked so
      the LHS dataset builder can verify the design doc's
      "< 1 % of cells clamp" assumption.
    """

    t_s: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    position_m: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    state_of_charge: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    power_in_w: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    power_out_w: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    mobility_power_w: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    slip: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    sinkage_m: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    wheel_torque_nm: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    sun_elevation_deg: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    terminated_reason: str = ""
    battery_floored: bool = False
    rover_stalled: bool = False
    reached_distance: bool = False
    cruise_speed_mps: float = 0.0
    effective_duty_cycle: float = 0.0
    cruise_kinematic_clamped: bool = False
    peak_torque_demand_nm: float = 0.0
    peak_torque_capacity_nm: float = 0.0


# ---------------------------------------------------------------------------
# Per-step physics
# ---------------------------------------------------------------------------


def _required_dp_per_wheel_n(
    total_mass_kg: float,
    n_wheels: int,
    slope_deg: float,
    gravity_m_per_s2: float,
) -> float:
    """Drawbar-pull per wheel needed to sustain motion up a slope.

    Compaction / rolling resistance is already absorbed into the
    Bekker-Wong DP; only the gradient term appears here.
    """
    theta = math.radians(slope_deg)
    weight_n = total_mass_kg * gravity_m_per_s2
    return weight_n * math.sin(theta) / n_wheels


def _load_per_wheel_n(
    total_mass_kg: float,
    n_wheels: int,
    slope_deg: float,
    gravity_m_per_s2: float,
) -> float:
    """Normal load per wheel on a slope (cos(theta) projection)."""
    theta = math.radians(slope_deg)
    return total_mass_kg * gravity_m_per_s2 * math.cos(theta) / n_wheels


def _build_corr_feature_buffer(
    wheel: WheelGeometry,
    soil: SoilParameters,
    load_per_wheel_n: float,
) -> NDArray[np.float64]:
    """Pre-allocate the 12-d wheel-level feature buffer for one step.

    Order **must** match
    :data:`roverdevkit.terramechanics.correction_model.FEATURE_COLUMNS`
    (asserted at module import). Slip is left at 0.0 here; the brentq
    residual mutates ``buf[_CORR_SLIP_IDX]`` per call so each correction
    prediction only re-indexes one column instead of rebuilding the
    whole vector.
    """
    buf = np.empty(len(_CORR_FEATURE_COLUMNS), dtype=np.float64)
    buf[0] = load_per_wheel_n
    buf[1] = 0.0  # slip — overwritten per residual call
    buf[2] = wheel.radius_m
    buf[3] = wheel.width_m
    buf[4] = wheel.grouser_height_m
    buf[5] = float(wheel.grouser_count)
    buf[6] = soil.n
    buf[7] = soil.k_c
    buf[8] = soil.k_phi
    buf[9] = soil.cohesion_kpa
    buf[10] = soil.friction_angle_deg
    buf[11] = soil.shear_modulus_k_m
    return buf


def _corrected_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    load_per_wheel_n: float,
    slip: float,
    correction: WheelLevelCorrection | None,
    feature_buffer: NDArray[np.float64] | None,
) -> WheelForces:
    """Bekker-Wong forces, optionally augmented by the SCM correction.

    When ``correction is None`` returns the BW result with no overhead
    (this is the production analytical path). When a correction is
    provided, predicts ``(Δ DP, Δ τ, Δ sinkage)`` from the wheel-level
    feature buffer and adds them to the BW outputs. Sinkage is clipped
    at zero — BW + correction can occasionally go slightly negative
    near low-load / smooth-tire operating points and a non-physical
    negative sinkage would propagate confusingly into the traverse log.
    Slip is invariant: the correction lives in force-space at a given
    operating point, not slip-space.
    """
    bw = single_wheel_forces(wheel, soil, load_per_wheel_n, slip)
    if correction is None:
        return bw

    if feature_buffer is None:
        feature_buffer = _build_corr_feature_buffer(wheel, soil, load_per_wheel_n)
    feature_buffer[_CORR_SLIP_IDX] = slip
    deltas = correction.predict_array(feature_buffer)[0]
    # Use dataclasses.replace so any non-corrected WheelForces fields
    # (rolling_resistance_n, entry_angle_rad, …) are preserved verbatim
    # without us having to know the full schema here.
    return dataclasses.replace(
        bw,
        drawbar_pull_n=bw.drawbar_pull_n + float(deltas[0]),
        driving_torque_nm=bw.driving_torque_nm + float(deltas[1]),
        sinkage_m=max(0.0, bw.sinkage_m + float(deltas[2])),
    )


def _solve_step_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    load_per_wheel_n: float,
    required_dp_per_wheel_n: float,
    *,
    correction: WheelLevelCorrection | None = None,
) -> tuple[WheelForces, bool]:
    """Find the slip that balances DP(s) = required; return (forces, stalled).

    With ``correction is None`` this is pure Bekker-Wong; with a
    correction, the slip-balance equilibrium is solved against
    ``DP_BW(s) + ΔDP(s) − DP_required = 0`` so the resulting slip
    reflects corrected mobility (the §6 W7 plan calls this "injection
    point 1"; the corrected torque it returns is "injection point 2"
    once it propagates into ``_mobility_power_w``).

    If no slip in the bracket achieves the required corrected DP (e.g.
    slope too steep for this wheel-soil-correction combo) we pin slip
    at the upper bracket and flag ``stalled = True``.
    """
    feature_buffer = (
        _build_corr_feature_buffer(wheel, soil, load_per_wheel_n)
        if correction is not None
        else None
    )

    def residual(slip: float) -> float:
        return (
            _corrected_wheel_forces(
                wheel, soil, load_per_wheel_n, slip, correction, feature_buffer
            ).drawbar_pull_n
            - required_dp_per_wheel_n
        )

    r_low = residual(_SLIP_LOWER_BOUND)
    r_high = residual(_SLIP_UPPER_BOUND)

    if r_low > 0.0 and r_high > 0.0:
        # Surplus DP even at the lowest slip; operate at slip = 0.
        forces = _corrected_wheel_forces(
            wheel, soil, load_per_wheel_n, 0.0, correction, feature_buffer
        )
        return forces, False
    if r_low < 0.0 and r_high < 0.0:
        # Even at max slip we can't deliver required DP → stalled.
        forces = _corrected_wheel_forces(
            wheel, soil, load_per_wheel_n, _SLIP_UPPER_BOUND, correction, feature_buffer
        )
        return forces, True

    slip = float(brentq(residual, _SLIP_LOWER_BOUND, _SLIP_UPPER_BOUND, xtol=1e-4))
    forces = _corrected_wheel_forces(
        wheel, soil, load_per_wheel_n, slip, correction, feature_buffer
    )
    return forces, False


def _scm_solve_step_wheel_forces(
    wheel: WheelGeometry,
    soil: SoilParameters,
    load_per_wheel_n: float,
    required_dp_per_wheel_n: float,
) -> tuple[WheelForces, bool]:
    """SCM-direct counterpart of :func:`_solve_step_wheel_forces`.

    Solves the same slip-balance equation, but evaluates drawbar pull
    via the PyChrono SCM single-wheel driver instead of Bekker-Wong.
    Used by the Week-7.7 bake-off (BW vs BW+correction vs SCM-direct)
    and by the SCM-direct backend in :func:`run_traverse`.

    The PyChrono import is deferred so the analytical path never pays
    the ~350 ms OpenMP preload cost. Brentq is run at a coarser
    tolerance (``xtol=2e-3``) than the BW/BW+correction path because
    each residual evaluation costs ~0.1 s of physics simulation; the
    bracket [-0.9, 0.95] gives ~9-12 evaluations rather than ~17.
    """
    from roverdevkit.terramechanics.pychrono_scm import single_wheel_forces_scm

    def residual(slip: float) -> float:
        return (
            single_wheel_forces_scm(wheel, soil, load_per_wheel_n, slip).drawbar_pull_n
            - required_dp_per_wheel_n
        )

    r_low = residual(_SLIP_LOWER_BOUND)
    r_high = residual(_SLIP_UPPER_BOUND)

    if r_low > 0.0 and r_high > 0.0:
        return single_wheel_forces_scm(wheel, soil, load_per_wheel_n, 0.0), False
    if r_low < 0.0 and r_high < 0.0:
        return (
            single_wheel_forces_scm(wheel, soil, load_per_wheel_n, _SLIP_UPPER_BOUND),
            True,
        )

    slip = float(brentq(residual, _SLIP_LOWER_BOUND, _SLIP_UPPER_BOUND, xtol=2e-3))
    return single_wheel_forces_scm(wheel, soil, load_per_wheel_n, slip), False


def _mobility_power_w(
    forces: WheelForces,
    cruise_speed_mps: float,
    wheel_radius_m: float,
    n_wheels: int,
    motor_efficiency: float,
    stalled: bool,
) -> float:
    """Instantaneous electrical motor power to drive the rover at ``v``.

    Mechanical power per wheel is ``T * omega``, where the slip kinematic
    gives ``omega = v / (R * (1 - s))``. When the rover is stalled the
    motor still draws torque * omega at the no-forward-progress slip --
    the wheels are still spinning, just not pulling the rover forward.

    Schema v6 (W12 step B): ``cruise_speed_mps`` is now the *derived*
    rover speed from :func:`roverdevkit.drivetrain.motor.cruise_speed`,
    not the pre-v6 design input ``nominal_speed_mps``.
    """
    slip = forces.slip
    omega = cruise_speed_mps / (wheel_radius_m * max(1e-3, 1.0 - slip))
    mechanical_power_per_wheel = forces.driving_torque_nm * omega
    electrical_power_per_wheel = mechanical_power_per_wheel / max(1e-3, motor_efficiency)
    # If stalled, the rover still commands the wheels but makes no
    # headway; electrical draw is unchanged because torque and slip
    # both saturate at the upper bracket.
    _ = stalled
    return n_wheels * electrical_power_per_wheel


def _average_solar_power_w(
    *,
    scenario: MissionScenario,
    panel_area_m2: float,
    panel_efficiency: float,
    panel_dust_factor: float,
    panel_tilt_deg: float,
    panel_azimuth_deg: float,
    declination_deg: float,
    noon_hour_offset: float,
    n_samples: int = 200,
) -> float:
    """Mean solar input over the mission window, in W.

    Schema v6 (W12 step B). Used by the energy-balance cruise-speed
    solve in :func:`roverdevkit.drivetrain.motor.energy_balance_v_cruise`.
    Sampled (not integrated analytically) because the existing
    :func:`panel_power_w` already encodes the diurnal / polar
    geometry, and a flat 200-sample mean over the mission window costs
    < 1 ms per evaluation — well below the per-mission budget. Number
    of samples is overridable for tests.
    """
    if scenario.mission_duration_earth_days <= 0.0:
        return 0.0
    duration_s = scenario.mission_duration_earth_days * 24.0 * 3600.0
    t_arr = np.linspace(0.0, duration_s, n_samples)
    p_arr = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        t_hours = t_arr[k] / 3600.0
        hour_angle = lunar_hour_angle_deg(t_hours, noon_hour=noon_hour_offset)
        elev = sun_elevation_deg(
            scenario.latitude_deg, hour_angle, declination_deg=declination_deg
        )
        if panel_tilt_deg == 0.0:
            sun_az = 180.0
        else:
            sun_az = sun_azimuth_deg(
                scenario.latitude_deg,
                hour_angle,
                declination_deg=declination_deg,
            )
        p_arr[k] = panel_power_w(
            panel_area_m2=panel_area_m2,
            panel_efficiency=panel_efficiency,
            sun_elevation_deg=elev,
            panel_tilt_deg=panel_tilt_deg,
            panel_azimuth_deg=panel_azimuth_deg,
            sun_azimuth_deg=sun_az,
            dust_degradation_factor=panel_dust_factor,
        )
    return float(np.mean(p_arr))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_traverse(
    design: DesignVector,
    scenario: MissionScenario,
    soil: SoilParameters,
    total_mass_kg: float,
    *,
    dt_s: float = DEFAULT_DT_S,
    motor_efficiency: float = DEFAULT_MOTOR_EFFICIENCY,
    panel_efficiency: float = DEFAULT_PANEL_EFFICIENCY,
    panel_dust_factor: float = DEFAULT_PANEL_DUST_FACTOR,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
    initial_soc: float = 1.0,
    battery_min_soc: float = 0.15,
    gravity_m_per_s2: float = 1.625,
    declination_deg: float = 0.0,
    noon_hour_offset: float = LUNAR_SYNODIC_DAY_HOURS / 4.0,
    correction: WheelLevelCorrection | None = None,
    force_backend: str = "bw",
    operational_duty_cycle_override: float | None = None,
) -> TraverseLog:
    """March the rover through the scenario and return a full traverse log.

    The simulator always runs for the full ``mission_duration_earth_days``.
    Failure modes (battery floored, rover stalled, distance reached) are
    captured as log fields rather than early returns.

    Parameters
    ----------
    design
        12-D design vector (:mod:`roverdevkit.schema`).
    scenario
        Mission scenario (already validated / loaded from YAML).
    soil
        Bekker-Wong soil parameters for the scenario's ``soil_simulant``.
    total_mass_kg
        Vehicle mass from :mod:`roverdevkit.mass`.
    dt_s, motor_efficiency, panel_efficiency, panel_dust_factor
        Simulator knobs with project-plan defaults (see module constants).
    panel_tilt_deg, panel_azimuth_deg
        Geometry of the rover's solar array. Default is a horizontal
        top-mounted panel.
    initial_soc
        Battery state-of-charge at t=0. Default 1.0 (fully charged at
        mission start).
    battery_min_soc
        DoD floor forwarded to :class:`BatteryState`.
    gravity_m_per_s2
        Surface gravity (default lunar).
    declination_deg, noon_hour_offset
        Sun geometry controls; see :mod:`roverdevkit.power.solar`.
    correction
        Optional wheel-level SCM correction artifact. When provided
        (and ``force_backend != "scm"``), BW outputs are augmented by
        ``correction.predict_array(...)`` at the slip-balance solve.
        ``None`` (default) preserves the analytical BW-only path.
    force_backend
        Wheel-level force backend selector for the Week-7.7 bake-off:
        ``"bw"`` (default) uses Bekker-Wong, optionally augmented by
        ``correction``; ``"scm"`` calls the PyChrono SCM single-wheel
        driver directly inside the slip solve. ``"scm"`` ignores
        ``correction``. The wheel-force solve is loop-invariant in the
        current scenario schema and is computed once per mission, so
        the SCM-direct path costs ~1-3 s/mission rather than the
        ~10 minutes a per-step solve would take.
    operational_duty_cycle_override
        Schema v6 (W12 step B): per-call override of the scenario's
        ``operational_duty_cycle``. ``None`` (default) uses the value
        on the scenario YAML. Schema v7 (W12 step B follow-up): this
        value is used directly as ``δ_eff`` (clamped to ``[0, 1]``);
        the v6 ``min(δ_des, δ_ops)`` cap collapsed when
        ``designed_duty_cycle`` was removed from the design vector.
    """
    if force_backend not in {"bw", "scm"}:
        raise ValueError(f"force_backend must be 'bw' or 'scm', got {force_backend!r}")
    wheel = WheelGeometry(
        radius_m=design.wheel_radius_m,
        width_m=design.wheel_width_m,
        grouser_height_m=design.grouser_height_m,
        grouser_count=design.grouser_count,
    )
    load_per_wheel = _load_per_wheel_n(
        total_mass_kg, design.n_wheels, scenario.max_slope_deg, gravity_m_per_s2
    )
    required_dp_per_wheel = _required_dp_per_wheel_n(
        total_mass_kg, design.n_wheels, scenario.max_slope_deg, gravity_m_per_s2
    )

    battery = BatteryState(
        capacity_wh=design.battery_capacity_wh,
        state_of_charge=initial_soc,
        min_state_of_charge=battery_min_soc,
    )

    duration_s = scenario.mission_duration_earth_days * 24.0 * 3600.0
    n_steps = max(2, int(math.ceil(duration_s / dt_s)) + 1)
    t_arr = np.linspace(0.0, duration_s, n_steps)

    pos_arr = np.zeros(n_steps)
    soc_arr = np.zeros(n_steps)
    power_in_arr = np.zeros(n_steps)
    power_out_arr = np.zeros(n_steps)
    mobility_arr = np.zeros(n_steps)
    slip_arr = np.zeros(n_steps)
    sinkage_arr = np.zeros(n_steps)
    torque_arr = np.zeros(n_steps)
    elev_arr = np.zeros(n_steps)

    reached_distance = False
    battery_floored_once = False
    position = 0.0
    soc_arr[0] = battery.state_of_charge

    # Per-mission constants (lifted from the inner loop because every
    # input to _solve_step_wheel_forces and _mobility_power_w is
    # loop-invariant in the current scenario schema: wheel/soil/mass/
    # max_slope_deg are per-mission, not per-position. The inner loop
    # below only updates the time-variant state — sun geometry, solar
    # power, battery SOC, and rover position. If a future schema adds a
    # per-position slope or per-segment soil profile, this lift will
    # need to be reverted (or made conditional on the new fields).
    if force_backend == "scm":
        forces, slip_solver_failed = _scm_solve_step_wheel_forces(
            wheel, soil, load_per_wheel, required_dp_per_wheel
        )
    else:
        forces, slip_solver_failed = _solve_step_wheel_forces(
            wheel, soil, load_per_wheel, required_dp_per_wheel, correction=correction
        )

    # Schema v6/v7 (W12 step B): derive δ_eff and v_cruise here,
    # replacing the pre-v6 ``design.nominal_speed_mps`` and
    # ``design.drive_duty_cycle`` design inputs. The torque demand
    # from the slip-balance solve plus the mission-average solar budget
    # feed :func:`roverdevkit.drivetrain.motor.cruise_speed`.
    # ``stalled`` is composed inside that helper from "slip solver
    # failed" *or* "torque demand exceeds peak_wheel_torque_nm
    # capacity". Schema v7 collapsed the v6 ``min(δ_des, δ_ops)``
    # rule into a single per-scenario ``operational_duty_cycle`` after
    # that field turned out to do no engineering work in the v6 mass
    # model.
    ops_duty_used = (
        scenario.operational_duty_cycle
        if operational_duty_cycle_override is None
        else operational_duty_cycle_override
    )
    p_solar_avg_w = _average_solar_power_w(
        scenario=scenario,
        panel_area_m2=design.solar_area_m2,
        panel_efficiency=panel_efficiency,
        panel_dust_factor=panel_dust_factor,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
        declination_deg=declination_deg,
        noon_hour_offset=noon_hour_offset,
    )
    delta_eff = effective_duty_cycle(ops_duty_used)
    cruise: CruiseResult = cruise_speed(
        peak_wheel_torque_nm=design.peak_wheel_torque_nm,
        t_req_per_wheel_nm=float(forces.driving_torque_nm),
        slip_eq=float(forces.slip),
        slip_solver_failed=slip_solver_failed,
        p_solar_avg_w=p_solar_avg_w,
        p_avionics_w=design.avionics_power_w,
        wheel_radius_m=design.wheel_radius_m,
        motor_efficiency=motor_efficiency,
        delta_eff=delta_eff,
        n_wheels=design.n_wheels,
    )
    stalled = cruise.stalled
    v_cruise = cruise.v_cruise_mps

    p_drive = _mobility_power_w(
        forces,
        v_cruise,
        design.wheel_radius_m,
        design.n_wheels,
        motor_efficiency,
        stalled,
    )
    effective_mobility_w = delta_eff * p_drive
    dx_per_step = 0.0 if stalled else v_cruise * dt_s * delta_eff
    rover_stalled_once = stalled

    for k in range(1, n_steps):
        t_s = t_arr[k]
        t_hours = t_s / 3600.0

        # Solar power in: solar geom at this instant (the only
        # per-step physics call still inside the loop).
        hour_angle = lunar_hour_angle_deg(t_hours, noon_hour=noon_hour_offset)
        elev = sun_elevation_deg(scenario.latitude_deg, hour_angle, declination_deg=declination_deg)
        if panel_tilt_deg == 0.0:
            sun_az = 180.0  # unused for horizontal panel
        else:
            sun_az = sun_azimuth_deg(
                scenario.latitude_deg, hour_angle, declination_deg=declination_deg
            )
        p_solar = panel_power_w(
            panel_area_m2=design.solar_area_m2,
            panel_efficiency=panel_efficiency,
            sun_elevation_deg=elev,
            panel_tilt_deg=panel_tilt_deg,
            panel_azimuth_deg=panel_azimuth_deg,
            sun_azimuth_deg=sun_az,
            dust_degradation_factor=panel_dust_factor,
        )

        # Energy-feasibility throttle (W12 Step A, 2026-04-27).
        # When entering this step the battery is already at its floor
        # and the *unthrottled* power balance would be negative, the
        # rover physically cannot sustain commanded duty: it must drop
        # to whatever fraction of its design duty solar can support in
        # real time. Without this throttle the simulator would happily
        # report full forward progress while quietly violating the
        # battery floor for the rest of the mission, which made
        # range_km a capability envelope rather than an achievable
        # distance. See reports/week12_design/decision.md and the
        # 2026-04-27 project_log entry.
        p_load_full = design.avionics_power_w + effective_mobility_w
        p_net_full = p_solar - p_load_full
        floored_at_step_start = (
            battery.state_of_charge <= battery.min_state_of_charge + 1e-9
            and p_net_full < 0.0
        )
        if floored_at_step_start:
            p_mob_avail_w = max(0.0, p_solar - design.avionics_power_w)
            duty_throttled = min(
                delta_eff,
                p_mob_avail_w / max(p_drive, 1e-9),
            )
            dx = 0.0 if stalled else v_cruise * dt_s * duty_throttled
            p_load = design.avionics_power_w + duty_throttled * p_drive
            battery_floored_once = True
        else:
            dx = dx_per_step
            p_load = p_load_full

        # Forward progress for the step (post-throttle).
        remaining = scenario.traverse_distance_m - position
        if dx >= remaining:
            dx = max(0.0, remaining)
            reached_distance = True
        position += dx

        # Power balance and battery update.
        p_net = p_solar - p_load
        battery = battery_step(battery, p_net, dt_s)
        if battery.state_of_charge <= battery.min_state_of_charge + 1e-9 and p_net < 0.0:
            battery_floored_once = True

        pos_arr[k] = position
        soc_arr[k] = battery.state_of_charge
        power_in_arr[k] = p_solar
        power_out_arr[k] = p_load
        # Log the actual mobility draw (post-throttle), not the
        # hypothetical full-duty draw, so downstream diagnostics see
        # the real per-step power profile.
        mobility_arr[k] = max(0.0, p_load - design.avionics_power_w)
        slip_arr[k] = forces.slip
        sinkage_arr[k] = forces.sinkage_m
        torque_arr[k] = forces.driving_torque_nm
        elev_arr[k] = elev

    # Compose the termination message from the observed events.
    reasons: list[str] = []
    if reached_distance:
        reasons.append("traverse distance reached")
    if battery_floored_once:
        reasons.append("battery hit SOC floor at least once")
    if rover_stalled_once:
        reasons.append("rover stalled on slope at least once")
    if not reasons:
        reasons.append("mission duration elapsed nominally")

    return TraverseLog(
        t_s=t_arr,
        position_m=pos_arr,
        state_of_charge=soc_arr,
        power_in_w=power_in_arr,
        power_out_w=power_out_arr,
        mobility_power_w=mobility_arr,
        slip=slip_arr,
        sinkage_m=sinkage_arr,
        wheel_torque_nm=torque_arr,
        sun_elevation_deg=elev_arr,
        terminated_reason="; ".join(reasons),
        battery_floored=battery_floored_once,
        rover_stalled=rover_stalled_once,
        reached_distance=reached_distance,
        cruise_speed_mps=float(v_cruise),
        effective_duty_cycle=float(delta_eff),
        cruise_kinematic_clamped=bool(cruise.kinematic_clamped),
        peak_torque_demand_nm=float(forces.driving_torque_nm),
        peak_torque_capacity_nm=float(design.peak_wheel_torque_nm),
    )
