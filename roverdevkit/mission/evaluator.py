"""Top-level mission evaluator.

This is the **primary artifact** of the package. After
the traverse-loop lift-out it runs in ~30 ms / mission on the
analytical Bekker-Wong path; the
:mod:`roverdevkit.surrogate` layer is an *optional* acceleration and
uncertainty layer used for NSGA-II inner loops, batch sensitivity studies,
and prediction-interval calibration. Most webapp workflows can run against
this evaluator directly. Every ML claim in the paper is grounded in what
this function computes.

Capability envelope vs operational utilisation
----------------------------------------------
Schema v6 (v6 schema update) introduced an explicit
engineering-vs-operations duty-cycle split (``designed_duty_cycle``
on the design vector vs ``operational_duty_cycle`` on the scenario,
with the evaluator running the loop at ``δ_eff = min(δ_des, δ_ops)``)
to match the same distinction JPL Team X and ESA CDF studies use.
Schema v7 (v7 schema follow-up) collapsed that split back into a
single per-scenario ``operational_duty_cycle`` after
``designed_duty_cycle`` turned out to do no engineering work in the
v6 mass model — the only role of ``δ_des`` was to upper-bound
``δ_eff``, which a user can equivalently express by lowering
``operational_duty_cycle``. The pre-v6 ``range_at_utilisation``
post-hoc rescaler remains gone; per-call ops duty is exposed via the
``operational_duty_cycle`` override on :func:`evaluate`. Calibrated
defaults follow published ground-ops cadence (mare 0.30, crater 0.20,
highland 0.15, polar 0.05).

Pipeline
--------
1. Mass model  -> total vehicle mass + per-subsystem breakdown
   (:mod:`roverdevkit.mass`).
2. Thermal     -> binary survive-the-mission flag
   (:mod:`roverdevkit.power.thermal`).
3. Soil lookup -> Bekker-Wong parameters for the scenario's simulant
   (:mod:`roverdevkit.terramechanics.soils`).
4. Capability  -> max climbable slope on this soil
   (:mod:`roverdevkit.mission.capability`).
5. Traverse    -> time-stepped run-to-completion log
   (:mod:`roverdevkit.mission.traverse_sim`).
6. Aggregate   -> MissionMetrics (schema).

Public API::

    from roverdevkit.mission.evaluator import evaluate
    from roverdevkit.mission.scenarios import load_scenario
    from roverdevkit.schema import DesignVector

    metrics = evaluate(design, load_scenario("equatorial_mare_traverse"))

Design notes
------------
- The evaluator **always returns** a :class:`MissionMetrics` object; it
  does not short-circuit on design failures. Constraint flags
  (``thermal_survival``, ``stalled``) and continuous metrics
  (``energy_margin_pct``, ``range_km``) encode the failure modes instead.
  This is critical for training the surrogate-training surrogate over the full
  design space including infeasible regions.
- Schema v6 (v6 schema update): ``stalled`` replaces the v5 ``motor_torque_ok``
  field. The stall gate is now an explicit comparison against
  :attr:`roverdevkit.schema.DesignVector.peak_wheel_torque_nm` — the
  drivetrain stalls when the slip-balance torque demand exceeds that
  capacity, or when the slip solver could not develop the required
  drawbar pull. See :mod:`roverdevkit.drivetrain.motor`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from roverdevkit.mass.parametric_mers import (
    MassBreakdown,
    MassModelParams,
    estimate_mass_from_design,
)
from roverdevkit.mission.capability import max_climbable_slope_deg
from roverdevkit.mission.traverse_sim import TraverseLog, run_traverse
from roverdevkit.power.thermal import (
    ThermalArchitecture,
    ThermalResult,
    default_architecture_for_design,
    evaluate_thermal,
)
from roverdevkit.schema import DesignVector, MissionMetrics, MissionScenario
from roverdevkit.terramechanics.bekker_wong import SoilParameters, WheelGeometry
from roverdevkit.terramechanics.soils import get_soil_parameters


@dataclass(frozen=True)
class DetailedEvaluation:
    """Full evaluator output: headline metrics plus supporting artefacts.

    Returned by :func:`evaluate_verbose`. surrogate-training dataset generation needs
    the :class:`TraverseLog` so it can compute aggregate sub-model stats
    (peak/mean/p95 of drawbar pull, sinkage, motor torque, solar power,
    battery SOC) that the single-scalar :class:`MissionMetrics` does not
    expose. The :class:`MassBreakdown` is kept alongside so per-subsystem
    mass is recoverable without re-running the mass model. The full
    :class:`ThermalResult` is also surfaced (webapp web app reads
    peak / cold temperatures so the constraint chip can explain *why*
    a survival flag fired).
    """

    metrics: MissionMetrics
    log: TraverseLog
    mass: MassBreakdown
    thermal: ThermalResult


def _energy_margin_pct(log: TraverseLog, min_soc: float) -> float:
    """Discretionary-energy margin at end of mission, percent.

    0 % = battery sitting on the DoD floor; 100 % = full charge above
    the floor. Defined as ``(SOC_end - min_SOC) / (1 - min_SOC) * 100``
    with a clamp at 0 so unsurvivable missions return 0 rather than a
    negative number.

    This is the **reporting** metric (clipped, monotonically interpretable).
    For the surrogate-training signal that does not saturate at 0/100, see
    :func:`_energy_margin_raw_pct`.
    """
    if log.state_of_charge.size == 0:
        return 0.0
    soc_end = float(log.state_of_charge[-1])
    span = max(1e-9, 1.0 - min_soc)
    return max(0.0, (soc_end - min_soc) / span * 100.0)


def _energy_margin_raw_pct(log: TraverseLog) -> float:
    """Mission-integrated energy balance as a percentage of consumption.

    Defined as ``(E_generated - E_consumed) / E_consumed * 100``,
    unbounded on both sides. Negative ⇒ net energy deficit; >0 ⇒ surplus
    generation. Used by the surrogate-training surrogate because it does not
    saturate when SOC sits at 1.0 (benign scenarios) or at the DoD floor
    (polar night), unlike :func:`_energy_margin_pct`.

    Computed via trapezoidal integration of the traverse log's
    ``power_in_w`` (solar input) and ``power_out_w`` (avionics +
    mobility). Time is assumed monotonic and in seconds.
    """
    if log.t_s.size < 2:
        return 0.0
    t = log.t_s
    e_in_wh = float(np.trapezoid(log.power_in_w, t)) / 3600.0
    e_out_wh = float(np.trapezoid(log.power_out_w, t)) / 3600.0
    if e_out_wh <= 1e-9:
        return 0.0
    return (e_in_wh - e_out_wh) / e_out_wh * 100.0


def evaluate_verbose(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    mass_params: MassModelParams | None = None,
    thermal_architecture: ThermalArchitecture | None = None,
    gravity_m_per_s2: float | None = None,
    soil_override: SoilParameters | None = None,
    operational_duty_cycle: float | None = None,
    payload_mass_kg: float | None = None,
    payload_power_w: float | None = None,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
) -> DetailedEvaluation:
    """Full evaluator: headline metrics plus traverse log and mass breakdown.

    Same physics pipeline as :func:`evaluate`, but returns the supporting
    artefacts needed by the surrogate-training dataset builder (aggregate sub-model
    statistics from the :class:`TraverseLog`) and per-subsystem mass
    introspection for validation.

    Parameters
    ----------
    design
        12-D design vector.
    scenario
        Mission context (latitude, terrain, distance, sun geometry).
    mass_params
        Optional :class:`MassModelParams` override.
    thermal_architecture
        Optional :class:`ThermalArchitecture` override. If ``None``, a
        default enclosure is built from a fraction of the chassis using
        :func:`default_architecture_for_design`.
    gravity_m_per_s2
        Surface gravity override (e.g. for off-Moon test scenarios).
        All current registry rovers run at lunar gravity since the
        Mars-gravity Sojourner sentinel was removed (2026-04-25).
    soil_override
        Optional :class:`SoilParameters` to use instead of the
        catalogue lookup on ``scenario.soil_simulant``. The surrogate-training
        LHS sweep uses this to inject per-sample jittered Bekker
        parameters so the surrogate learns a continuous soil → metric
        mapping instead of a four-category one
        used by the surrogate-training workflow.
    operational_duty_cycle
        Schema v6 (v6 schema update): per-call override of
        ``scenario.operational_duty_cycle``. ``None`` (default) uses
        the scenario YAML's calibrated value. Schema v7 (v6 schema update
        follow-up) uses this value directly as ``δ_eff`` (clamped to
        ``[0, 1]``); the v6 ``min(δ_des, δ_ops)`` cap collapsed when
        ``designed_duty_cycle`` was removed from the design vector.
    payload_mass_kg, payload_power_w
        Schema v9: per-call override of the scenario's
        ``payload_mass_kg`` / ``payload_power_w`` mission-requirement
        fields. ``None`` (default) uses the scenario YAML values.
        ``payload_mass_kg`` is added to total vehicle mass as a
        top-level line item outside the dry-mass growth margin;
        ``payload_power_w`` is added to the continuous ops-time
        electrical load (alongside avionics) and to the hot-case
        thermal dissipation. The rediscovery harness forwards a
        rover's published payload to both the rover re-evaluation and
        every NSGA-II individual so the comparison stays
        apples-to-apples.
    panel_tilt_deg, panel_azimuth_deg
        Solar-array orientation forwarded to
        :func:`roverdevkit.mission.traverse_sim.run_traverse`.
        Defaults match the simulator's historical horizontal /
        south-facing panel; pass non-zero values to model
        polar-deployable arrays whose surface normals track the
        low-elevation sun (see
        :class:`roverdevkit.validation.rover_registry.RoverRegistryEntry`
        for per-rover values, and the rediscovery harness for the
        scenario-driven ``tilt = min(80, |latitude|)`` override
        used at high latitudes).
    """
    mass_params = mass_params or MassModelParams()
    if gravity_m_per_s2 is not None and not math.isclose(
        gravity_m_per_s2, mass_params.gravity_moon_m_per_s2
    ):
        mass_params = dataclasses.replace(mass_params, gravity_moon_m_per_s2=gravity_m_per_s2)
    active_g = mass_params.gravity_moon_m_per_s2

    # Schema v9: resolve payload mission requirements (per-call override
    # falls back to the scenario default).
    payload_mass = (
        scenario.payload_mass_kg if payload_mass_kg is None else payload_mass_kg
    )
    payload_power = (
        scenario.payload_power_w if payload_power_w is None else payload_power_w
    )

    breakdown: MassBreakdown = estimate_mass_from_design(
        design, params=mass_params, payload_mass_kg=payload_mass
    )
    total_mass_kg = breakdown.total_kg

    if thermal_architecture is None:
        # Rough enclosure surface-area proxy: scales with chassis mass
        # via a cube-root law (box side ~ mass^(1/3) * density^(-1/3)).
        # 0.02 m^2/kg^(2/3) is a coarse calibration that gives ~0.07 m^2
        # for a 6 kg chassis and ~0.24 m^2 for a 30 kg chassis.
        surface_area_m2 = 0.02 * (design.chassis_mass_kg ** (2.0 / 3.0)) + 0.05
        thermal_architecture = default_architecture_for_design(surface_area_m2=surface_area_m2)
    thermal_result = evaluate_thermal(
        thermal_architecture,
        # Schema v9: payload power dissipates as heat in the hot case,
        # so it adds to the operating-mode internal load.
        design.avionics_power_w + payload_power,
        scenario.latitude_deg,
    )
    thermal_ok = thermal_result.survives

    soil = (
        soil_override if soil_override is not None else get_soil_parameters(scenario.soil_simulant)
    )

    wheel = WheelGeometry(
        radius_m=design.wheel_radius_m,
        width_m=design.wheel_width_m,
        grouser_height_m=design.grouser_height_m,
        grouser_count=design.grouser_count,
    )
    slope_capability = max_climbable_slope_deg(
        wheel,
        soil,
        total_mass_kg=total_mass_kg,
        n_wheels=design.n_wheels,
        gravity_m_per_s2=active_g,
    )

    log = run_traverse(
        design,
        scenario,
        soil,
        total_mass_kg=total_mass_kg,
        gravity_m_per_s2=active_g,
        operational_duty_cycle_override=operational_duty_cycle,
        payload_power_w=payload_power,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    )

    range_km = float(log.position_m[-1]) / 1000.0
    energy_margin_pct = _energy_margin_pct(log, min_soc=0.15)
    energy_margin_raw_pct = _energy_margin_raw_pct(log)
    peak_torque_nm = float(np.max(np.abs(log.wheel_torque_nm))) if log.wheel_torque_nm.size else 0.0
    sinkage_max_m = float(np.max(log.sinkage_m)) if log.sinkage_m.size else 0.0

    _ = active_g  # documents that gravity flows through mass_params above
    stalled = bool(log.rover_stalled)

    # Guard against NaN/inf creeping out of any sub-model; cap to safe
    # defaults so downstream pydantic validation always succeeds.
    if not math.isfinite(range_km):
        range_km = 0.0
    if not math.isfinite(energy_margin_pct):
        energy_margin_pct = 0.0
    if not math.isfinite(energy_margin_raw_pct):
        energy_margin_raw_pct = 0.0
    if not math.isfinite(peak_torque_nm):
        peak_torque_nm = 0.0
    if not math.isfinite(sinkage_max_m):
        sinkage_max_m = 0.0

    metrics = MissionMetrics(
        range_km=range_km,
        energy_margin_pct=energy_margin_pct,
        slope_capability_deg=slope_capability,
        energy_margin_raw_pct=energy_margin_raw_pct,
        total_mass_kg=total_mass_kg,
        peak_motor_torque_nm=peak_torque_nm,
        sinkage_max_m=sinkage_max_m,
        thermal_survival=thermal_ok,
        stalled=stalled,
    )
    return DetailedEvaluation(
        metrics=metrics, log=log, mass=breakdown, thermal=thermal_result
    )


def evaluate(
    design: DesignVector,
    scenario: MissionScenario,
    *,
    mass_params: MassModelParams | None = None,
    thermal_architecture: ThermalArchitecture | None = None,
    gravity_m_per_s2: float | None = None,
    soil_override: SoilParameters | None = None,
    operational_duty_cycle: float | None = None,
    payload_mass_kg: float | None = None,
    payload_power_w: float | None = None,
    panel_tilt_deg: float = 0.0,
    panel_azimuth_deg: float = 180.0,
) -> MissionMetrics:
    """Run the full mission evaluator on one design in one scenario.

    Thin wrapper around :func:`evaluate_verbose` that discards the
    :class:`TraverseLog` and :class:`MassBreakdown`. This is the
    canonical public entry point; callers that need the supporting
    artefacts (e.g. the surrogate-training dataset builder) should call
    ``evaluate_verbose`` directly.
    """
    return evaluate_verbose(
        design,
        scenario,
        mass_params=mass_params,
        thermal_architecture=thermal_architecture,
        gravity_m_per_s2=gravity_m_per_s2,
        soil_override=soil_override,
        operational_duty_cycle=operational_duty_cycle,
        payload_mass_kg=payload_mass_kg,
        payload_power_w=payload_power_w,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
    ).metrics
