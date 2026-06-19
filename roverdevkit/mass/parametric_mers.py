"""Bottom-up parametric mass model for lunar micro-rovers.

Approach
--------
Each subsystem mass
is computed from a **physics-grounded specific mass or a standard
spacecraft-sizing fraction** with a cited source. The rows in
``data/mass_validation_set.csv`` are then used as a **validation set**
(see :mod:`roverdevkit.mass.validation`) - "does the bottom-up model
reproduce total mass within ~30 % for each real rover?".

The model is deliberately transparent: every coefficient is exposed as a
field of :class:`MassModelParams` so it can be overridden for sensitivity
studies from the surrogate / tradespace layer. Default values are chosen
from published space-hardware sources; see each field's docstring for the
citation.

Subsystem accounting (SMAD Ch. 11, Table 11-43 convention)::

    m_subsystems = m_chassis + m_wheels + m_motors + m_solar + m_battery + m_avionics
    m_harness    = f_harness * m_subsystems
    m_thermal    = f_thermal * (m_subsystems + m_harness)
    m_dry        = m_subsystems + m_harness + m_thermal
    m_margin     = f_margin * m_dry
    m_total      = m_dry + m_margin + m_payload

Payload mass (schema v9). Scientific payload is a *mission
requirement* carried on :class:`roverdevkit.schema.MissionScenario`,
not a design variable. It enters the total as a top-level line item
**after** the AIAA S-120A dry-mass growth margin (``m_payload`` is a
known, specified mass, so the bus growth allowance does not apply to
it). This matches standard aerospace mass-budget practice (payload is
tracked separately from bus dry mass) and lets the bottom-up model
reproduce full-up published rover mass — e.g. Yutu-2's ~25 kg science
payload no longer has to be hidden inside ``chassis_mass_kg``.

Motor mass (schema v6, v6 schema update). The motor subsystem mass is now
computed directly from the design's
:attr:`roverdevkit.schema.DesignVector.peak_wheel_torque_nm` (a true
input), so the pre-v6 fixed-point loop over total mass is gone — this
function is now strictly bottom-up and converges in a single pass.
The pre-v6 implicit mass-derived torque ceiling lives on in
:func:`roverdevkit.drivetrain.motor.sizing_peak_torque_anchor_nm` only
as the LHS prior anchor.

Primary references
------------------
Larson, W. J. & Wertz, J. R. *Space Mission Analysis and Design (SMAD)*,
3rd ed., Microcosm/Springer, 1999.
    Ch. 11 Table 11-43 - subsystem mass fractions.
    Ch. 16 - C&DH MERs.

Larson, W. J. & Pranke, L. K. *Human Spaceflight: Mission Analysis and
Design*, McGraw-Hill, 2000. Surface-system sizing.

AIAA S-120A-2015 *Mass Properties Control for Space Systems*, dry-mass
growth allowances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from roverdevkit.architecture import (
    ArchitectureParams,
    MobilityArchitecture,
    architecture_suspension_mass_kg,
)
from roverdevkit.schema import DesignVector

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassModelParams:
    """Specific-mass constants and sizing fractions for the bottom-up model.

    All values are exposed so the tradespace layer can sweep them for
    sensitivity analysis. Defaults are cited in-field.
    """

    # -- Wheels -------------------------------------------------------------
    wheel_structural_area_density_kg_per_m2: float = 8.0
    """Mass per unit of wheel-side area (2*pi*R*W), kg/m^2.

    Covers rim, hub, spokes, and fastener hardware for aluminium/CFRP rigid
    wheels in the 0.05-0.25 m radius class. Default chosen for the
    micro-rover mass class where thin-gauge aluminium or composite wheels
    dominate. Tune upward toward 15 kg/m^2 for MER/MSL-style stiff-rim wheels.
    """

    grouser_plate_thickness_m: float = 0.002
    """Grouser-plate thickness, m. 2 mm aluminium is typical for
    micro-rover traction fins (Bauer et al., i-SAIRAS 2005; MER grouser
    geometry scaled to micro-rover class)."""

    grouser_material_density_kg_per_m3: float = 2700.0
    """Grouser plate material density, kg/m^3. Default = 6061-T6 Al."""

    # -- Motors and drives -------------------------------------------------
    motor_base_mass_kg: float = 0.15
    """Irreducible motor + gearbox housing mass per wheel, kg.
    Floor for small brushless motors (~20-50 W) paired with a compact
    planetary or harmonic-drive reducer. Maxon EC-i 32 + GP 32 reaches
    ~0.12 kg; we round up to 0.15 kg to cover space-qualified bearings,
    shaft seals, and a flight-heritage connector."""

    motor_specific_torque_kg_per_nm: float = 0.10
    """Mass per unit of peak output (post-gearbox) torque, kg/(N*m).

    Calibrated against vendor catalogues: Maxon EC-i 32 + GP 32 AR
    planetary (100:1) = 0.325 kg at 4 N*m peak output -> 0.08 kg/(N*m);
    Maxon EC-i 40 + GP 52 (80:1) = 1.15 kg at ~20 N*m peak output ->
    0.06 kg/(N*m). We use 0.10 kg/(N*m) as a slightly conservative
    centre of the 0.06-0.12 kg/(N*m) range. Applies to the output
    torque; the motor itself produces a small fraction of this after
    the gear reduction."""

    motor_peak_friction_coef: float = 0.7
    """Peak tractive friction coefficient — schema v6 dead parameter.

    Pre-v6 the mass model sized motor torque internally from this
    coefficient and the rover's lunar weight. v6 makes
    :attr:`roverdevkit.schema.DesignVector.peak_wheel_torque_nm` a
    first-class design input, and this coefficient survives only as a
    default in
    :func:`roverdevkit.drivetrain.motor.sizing_peak_torque_anchor_nm`
    (the LHS prior anchor for the v6 dataset rebuild). Kept on
    :class:`MassModelParams` so existing callers / pickled fixtures
    don't break; remove on the next mass-model bump."""

    motor_sizing_safety_factor: float = 2.0
    """Schema v6 dead parameter — see :attr:`motor_peak_friction_coef`."""

    # -- Solar panels ------------------------------------------------------
    solar_specific_area_mass_kg_per_m2: float = 2.5
    """Areal mass density of a rigid body-mounted GaAs triple-junction solar
    panel including CFRP substrate and cell-to-substrate bond, kg/m^2.
    SMAD Table 11-43 gives 2.0-5.0 for body-mounted rigid panels;
    Spectrolab/AzurSpace datasheets for UTJ/ZTJ cells on a thin CFRP
    panel land near the lower bound."""

    # -- Battery -----------------------------------------------------------
    battery_pack_specific_energy_wh_per_kg: float = 120.0
    """Pack-level specific energy, Wh/kg. Li-ion cell-level ~200 Wh/kg
    multiplied by a ~0.6 pack-integration factor (BMS, casing, harness,
    thermal pads). NASA Glenn Battery Research Center tech reports;
    SMAD Ch. 11 secondary-battery table."""

    # -- Avionics and C&DH -------------------------------------------------
    avionics_base_mass_kg: float = 0.3
    """Floor mass for the smallest flyable avionics box, kg.
    Captures enclosure, backplane, and one CPU card. SMAD Ch. 16
    CDH MER lower bound."""

    avionics_specific_mass_kg_per_w: float = 0.05
    """Additional kg of structure / heat-sink per W of continuous avionics
    power dissipation. Derived from rule-of-thumb PCB-and-chassis thermal
    sizing at ~0.05 kg/W (SMAD Ch. 16)."""

    # -- Housekeeping fractions -------------------------------------------
    harness_fraction: float = 0.08
    """Harness mass as a fraction of the summed subsystem mass, SMAD
    Table 11-43 mid-range (6-10 %)."""

    thermal_fraction: float = 0.05
    """Thermal-control (MLI, heaters, straps) mass as a fraction of
    (subsystems + harness). SMAD Table 11-43 small-spacecraft mid-range
    (4-7 %)."""

    margin_fraction: float = 0.20
    """Dry-mass growth allowance (margin) as a fraction of dry mass.
    AIAA S-120A-2015 recommends 20 % at PDR maturity, dropping toward
    launch. Tradespace-level work uses the PDR number."""

    rocker_bogie_fixed_mass_kg: float = 0.5
    """Fixed rocker-bogie linkage / differential mass, kg."""

    rocker_bogie_chassis_fraction: float = 0.08
    """Additional rocker-bogie suspension mass as a fraction of chassis mass."""

    # -- Environment -------------------------------------------------------
    gravity_moon_m_per_s2: float = 1.625
    """Surface gravity at the lunar equator, m/s^2."""


# ---------------------------------------------------------------------------
# Breakdown container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassBreakdown:
    """Subsystem mass breakdown in kg. Sum of fields equals ``total_kg``."""

    chassis_kg: float
    wheels_kg: float
    motors_and_drives_kg: float
    solar_panels_kg: float
    battery_kg: float
    avionics_kg: float
    harness_kg: float
    thermal_kg: float
    margin_kg: float
    architecture_kg: float = 0.0
    payload_kg: float = 0.0
    """Scientific-payload mass, kg (schema v9).

    A mission requirement carried on
    :class:`roverdevkit.schema.MissionScenario`, added to the total
    *outside* the dry-mass growth margin. Defaults to 0.0 so pre-v9
    callers (and the mass model's own subsystem-only sweeps) are
    unaffected."""
    n_iterations: int = field(default=0, compare=False)
    """Number of fixed-point iterations taken to converge motor mass.

    Schema v6 (v6 schema update): always 1 — motor mass is a direct function
    of :attr:`roverdevkit.schema.DesignVector.peak_wheel_torque_nm` so
    the model converges in one pass. Field retained for back-compat
    with pre-v6 fixtures and the validation harness."""

    @property
    def total_kg(self) -> float:
        return (
            self.chassis_kg
            + self.wheels_kg
            + self.motors_and_drives_kg
            + self.solar_panels_kg
            + self.battery_kg
            + self.avionics_kg
            + self.harness_kg
            + self.thermal_kg
            + self.margin_kg
            + self.architecture_kg
            + self.payload_kg
        )

    @property
    def dry_kg(self) -> float:
        """Bus dry mass: excludes both the growth margin and the payload."""
        return self.total_kg - self.margin_kg - self.payload_kg


# ---------------------------------------------------------------------------
# Per-subsystem helpers
# ---------------------------------------------------------------------------


def _wheels_mass(
    wheel_radius_m: float,
    wheel_width_m: float,
    grouser_height_m: float,
    grouser_count: int,
    n_wheels: int,
    params: MassModelParams,
) -> float:
    """Rim-and-hub + grouser mass for all drive wheels.

    Structural term: ``rho_wheel_area * (2 * pi * R * W) * n_wheels``, where
    the side-area factor captures the dominant scaling of a rim-and-hub
    wheel with a thin cylindrical skin (calibrated to lunar-wheel
    hardware, not derived from first-principles shell theory).

    Grouser term: each grouser is modelled as a thin rectangular aluminium
    plate of dimensions ``W x h_g x t``; mass is
    ``N_g * W * h_g * t * rho_Al``.
    """
    if wheel_radius_m <= 0.0 or wheel_width_m <= 0.0 or n_wheels <= 0:
        raise ValueError("wheel_radius_m, wheel_width_m and n_wheels must be positive.")
    if grouser_height_m < 0.0 or grouser_count < 0:
        raise ValueError("grouser_height_m and grouser_count must be non-negative.")

    side_area_m2 = 2.0 * math.pi * wheel_radius_m * wheel_width_m
    structural_kg = params.wheel_structural_area_density_kg_per_m2 * side_area_m2

    grouser_volume_m3 = (
        grouser_count * wheel_width_m * grouser_height_m * params.grouser_plate_thickness_m
    )
    grouser_kg = grouser_volume_m3 * params.grouser_material_density_kg_per_m3

    return n_wheels * (structural_kg + grouser_kg)


def _motors_mass(
    n_wheels: int,
    peak_wheel_torque_nm: float,
    params: MassModelParams,
) -> float:
    """Drive-motor + gearbox mass sized from the peak-wheel torque.

    Schema v6 (v6 schema update): ``peak_wheel_torque_nm`` is now a direct
    design input rather than something derived from the vehicle's
    lunar weight inside the mass model. Per-motor mass remains
    ``m_0 + k_tau * tau_peak``; total summed over ``n_wheels``. The
    pre-v6 mass-derived ceiling lives on in
    :func:`roverdevkit.drivetrain.motor.sizing_peak_torque_anchor_nm`
    only as an LHS prior anchor.
    """
    if peak_wheel_torque_nm < 0.0:
        raise ValueError("peak_wheel_torque_nm must be non-negative.")
    if n_wheels <= 0:
        raise ValueError("n_wheels must be positive.")

    per_motor_kg = (
        params.motor_base_mass_kg
        + params.motor_specific_torque_kg_per_nm * peak_wheel_torque_nm
    )
    return n_wheels * per_motor_kg


def _solar_panels_mass(solar_area_m2: float, params: MassModelParams) -> float:
    if solar_area_m2 < 0.0:
        raise ValueError("solar_area_m2 must be non-negative.")
    return params.solar_specific_area_mass_kg_per_m2 * solar_area_m2


def _battery_mass(battery_capacity_wh: float, params: MassModelParams) -> float:
    if battery_capacity_wh < 0.0:
        raise ValueError("battery_capacity_wh must be non-negative.")
    return battery_capacity_wh / params.battery_pack_specific_energy_wh_per_kg


def _avionics_mass(avionics_power_w: float, params: MassModelParams) -> float:
    if avionics_power_w < 0.0:
        raise ValueError("avionics_power_w must be non-negative.")
    return params.avionics_base_mass_kg + params.avionics_specific_mass_kg_per_w * avionics_power_w


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def estimate_mass(
    *,
    wheel_radius_m: float,
    wheel_width_m: float,
    n_wheels: int,
    chassis_mass_kg: float,
    solar_area_m2: float,
    battery_capacity_wh: float,
    avionics_power_w: float,
    peak_wheel_torque_nm: float,
    grouser_height_m: float = 0.0,
    grouser_count: int = 0,
    payload_mass_kg: float = 0.0,
    mobility_architecture: MobilityArchitecture = "rigid_4wheel",
    params: MassModelParams | None = None,
) -> MassBreakdown:
    """Assemble a bottom-up subsystem mass breakdown for a rover design.

    Schema v6 (v6 schema update). All subsystems are load-independent now
    that ``peak_wheel_torque_nm`` is a true design input — the pre-v6
    fixed-point iteration over total mass is gone, and this function
    converges in a single pass. The ``n_iterations`` field on the
    returned :class:`MassBreakdown` is kept for backward compatibility
    but is always 1 in v6.

    The keyword-only signature matches the design-variable names on
    :class:`roverdevkit.schema.DesignVector`. See
    :func:`estimate_mass_from_design` for a convenience wrapper.

    Parameters
    ----------
    wheel_radius_m, wheel_width_m
        Wheel geometry, m.
    n_wheels
        Drive-wheel count (4 or 6 per :class:`DesignVector`).
    chassis_mass_kg
        Dry chassis structural mass, kg. A design-variable input.
    solar_area_m2, battery_capacity_wh, avionics_power_w
        Power-subsystem design variables.
    peak_wheel_torque_nm
        Peak per-wheel hub torque the drivetrain delivers, Nm. Sizes
        motor mass directly via ``m_0 + k_tau * tau_peak``.
    grouser_height_m, grouser_count
        Grouser geometry, m and count. Defaults to 0.
    payload_mass_kg
        Scientific-payload mass, kg (schema v9). A mission requirement
        from :attr:`roverdevkit.schema.MissionScenario.payload_mass_kg`.
        Added to the total *after* the dry-mass growth margin (payload
        is a known mass, not grown). Defaults to 0.0.
    params
        :class:`MassModelParams` override; defaults to the module defaults.

    Returns
    -------
    MassBreakdown
        Subsystem masses summing to the total vehicle mass.

    Raises
    ------
    ValueError
        On any non-physical input (negative masses, non-positive
        geometry).
    """
    params = params or MassModelParams()

    if payload_mass_kg < 0.0:
        raise ValueError("payload_mass_kg must be non-negative.")

    m_chassis = chassis_mass_kg
    if m_chassis <= 0.0:
        raise ValueError("chassis_mass_kg must be positive.")
    m_wheels = _wheels_mass(
        wheel_radius_m, wheel_width_m, grouser_height_m, grouser_count, n_wheels, params
    )
    m_solar = _solar_panels_mass(solar_area_m2, params)
    m_battery = _battery_mass(battery_capacity_wh, params)
    m_avionics = _avionics_mass(avionics_power_w, params)
    m_motors = _motors_mass(n_wheels, peak_wheel_torque_nm, params)

    m_subsystems = m_chassis + m_wheels + m_motors + m_solar + m_battery + m_avionics
    m_architecture = architecture_suspension_mass_kg(
        mobility_architecture,
        m_chassis,
        params=ArchitectureParams(
            rocker_bogie_fixed_mass_kg=params.rocker_bogie_fixed_mass_kg,
            rocker_bogie_chassis_fraction=params.rocker_bogie_chassis_fraction,
        ),
    )
    m_subsystems += m_architecture
    m_harness = params.harness_fraction * m_subsystems
    m_thermal = params.thermal_fraction * (m_subsystems + m_harness)
    m_dry = m_subsystems + m_harness + m_thermal
    m_margin = params.margin_fraction * m_dry

    return MassBreakdown(
        chassis_kg=m_chassis,
        wheels_kg=m_wheels,
        motors_and_drives_kg=m_motors,
        solar_panels_kg=m_solar,
        battery_kg=m_battery,
        avionics_kg=m_avionics,
        harness_kg=m_harness,
        thermal_kg=m_thermal,
        margin_kg=m_margin,
        architecture_kg=m_architecture,
        payload_kg=payload_mass_kg,
        n_iterations=1,
    )


def estimate_mass_from_design(
    design: DesignVector,
    params: MassModelParams | None = None,
    *,
    payload_mass_kg: float = 0.0,
) -> MassBreakdown:
    """Convenience wrapper that unpacks a :class:`DesignVector`.

    ``payload_mass_kg`` (schema v9) is a mission requirement that lives
    on :class:`roverdevkit.schema.MissionScenario`, not on the design
    vector, so it is passed in explicitly by the evaluator. Defaults to
    0.0 for callers that only need the bus mass.
    """
    return estimate_mass(
        wheel_radius_m=design.wheel_radius_m,
        wheel_width_m=design.wheel_width_m,
        n_wheels=design.n_wheels,
        chassis_mass_kg=design.chassis_mass_kg,
        solar_area_m2=design.solar_area_m2,
        battery_capacity_wh=design.battery_capacity_wh,
        avionics_power_w=design.avionics_power_w,
        peak_wheel_torque_nm=design.peak_wheel_torque_nm,
        grouser_height_m=design.grouser_height_m,
        grouser_count=design.grouser_count,
        payload_mass_kg=payload_mass_kg,
        mobility_architecture=design.mobility_architecture,
        params=params,
    )
