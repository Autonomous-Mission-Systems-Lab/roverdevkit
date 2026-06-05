"""Shared data schemas for design vectors, scenarios, and mission metrics.

These are the canonical types that flow between the mission evaluator,
surrogate, and tradespace layers. Using Pydantic gives us validation at the
boundaries (e.g. reject a wheel radius outside the design-space bounds) and
free JSON/YAML serialization for scenario config files.

Design-variable ranges are chosen to cover the 5-50 kg lunar micro-rover class and the public rover registry.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Design vector
# ---------------------------------------------------------------------------


class DesignVector(BaseModel):
    """A single point in the 11-dimensional rover design space.

    Units are SI unless otherwise noted.

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mobility
    wheel_radius_m: float = Field(ge=0.05, le=0.20, description="Wheel radius R")
    wheel_width_m: float = Field(
        ge=0.03,
        le=0.20,
        description=(
            "Wheel width W. Upper bound 0.20 m covers the heavier "
            "lunar-class micro-rovers (Yutu-2-class wheels are 0.15 m; "
            "Lunokhod-class is 0.20 m). Widened from 0.15 in the v3 "
            "LHS bounds widening to admit more representative validation "
            "rovers as in-distribution points."
        ),
    )
    grouser_height_m: float = Field(
        ge=0.0,
        le=0.020,
        description=(
            "Grouser height h_g. Upper bound 20 mm covers published lunar "
            "micro-rover wheels (Rashid-1 flew 15 mm, Yutu-class wheels "
            "use ~12 mm). The LHS sampler in surrogate.sampling currently "
            "draws to 12 mm only; widening it to the schema ceiling is a "
            "dataset-regeneration task for a future surrogate release."
        ),
    )
    grouser_count: int = Field(ge=0, le=24, description="Number of grousers N_g")
    n_wheels: Literal[4, 6] = Field(description="Wheel count N_w")

    # Chassis
    chassis_mass_kg: float = Field(
        ge=0.5,
        le=50.0,
        description=(
            "Dry chassis mass m_c. Upper bound 50 kg widened from 35 kg "
            "in the v3 LHS bounds widening so the heavier flown lunar "
            "micro-rovers (Yutu-class, ~30-40 kg ex-payload) sit inside "
            "the surrogate's training support rather than at a corner. "
            "Floor lowered from 3 kg to 0.5 kg (2026-05-27) to admit "
            "in-class flotilla / ultra-micro rovers (NASA JPL CADRE "
            "units ~0.8 kg chassis; iSpace Tenacious ~2 kg chassis). "
            "The v4 LHS dataset was sampled on the 3-50 kg range so "
            "designs below 3 kg are OOD for the existing surrogate "
            "until the v5 regeneration."
        ),
    )
    wheelbase_m: float = Field(ge=0.3, le=1.2, description="Wheelbase L_wb")

    # Power
    solar_area_m2: float = Field(ge=0.1, le=1.5, description="Solar array area A_s")
    battery_capacity_wh: float = Field(
        ge=5.0,
        le=500.0,
        description=(
            "Battery capacity C_b. Floor lowered from 20 Wh to 5 Wh "
            "(2026-05-27) to admit CADRE-class flotilla rovers "
            "(~10 Wh per unit). Designs below 20 Wh are OOD for the "
            "v4 surrogate until the v5 LHS regeneration."
        ),
    )
    avionics_power_w: float = Field(ge=5.0, le=40.0, description="Continuous avionics draw P_a")

    # Operations
    peak_wheel_torque_nm: float = Field(
        ge=0.05,
        le=20.0,
        description=(
            "Peak per-wheel hub torque T_hub^peak that the drivetrain "
            "(motor + gearbox combined) can sustain. Sizes motor mass via "
            "the parametric mass model and gates whether the rover stalls "
            "on slope. Cruise speed is *derived* from this, the slip-balance "
            "torque demand, and the steady-state power budget — not a free "
            "design variable. Bounds: 0.05 Nm captures CADRE-class "
            "flotilla rovers (2 kg / 4-wheel / R=0.08 at lunar gravity "
            "anchors at ~0.06 Nm per wheel); 20 Nm covers over-sized "
            "direct-drive concepts at the top of the design space. "
        ),
    )


# ---------------------------------------------------------------------------
# Mission scenario
# ---------------------------------------------------------------------------


TerrainClass = Literal["mare_nominal", "mare_loose", "highland_dense", "polar_regolith"]
ScenarioName = Literal[
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
]


class MissionScenario(BaseModel):
    """Fixed mission context against which a design is evaluated.

    Scenarios are typically loaded from YAML in
    :mod:`roverdevkit.mission.scenarios`. The four canonical scenarios
    (``ScenarioName``) are what the tradespace optimiser sweeps in
    webapp; validation scenarios (real-rover comparison harness)
    reuse the same schema with descriptive names, so ``name`` is a free
    string rather than the Literal. Invalid values never reach the
    optimiser because that path goes through ``load_scenario()``, which
    takes a ``ScenarioName`` Literal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    traverse_distance_m: float = Field(gt=0.0)
    terrain_class: TerrainClass
    soil_simulant: str = Field(
        description="Key into data/soil_simulants.csv, e.g. 'Apollo_regolith_nominal'."
    )
    mission_duration_earth_days: float = Field(gt=0.0)
    max_slope_deg: float = Field(ge=0.0, le=35.0, default=15.0)
    sun_geometry: Literal["continuous", "diurnal", "polar_intermittent"] = "diurnal"
    operational_duty_cycle: float = Field(
        ge=0.0,
        le=0.6,
        default=0.05,
        description=(
            "Calibrated against published "
            "rover-on-mission ops cadence (mare 0.30, crater 0.20, highland "
            "0.15, polar 0.05; see ``data/analytical/SCHEMA.md``). "
            "Users can override via the /evaluate / /predict API "
            "parameter; the surrogate is trained on δ_ops as an LHS "
            "feature (v7_1) so off-default queries keep calibrated PIs."
        ),
    )
    # Schema v9: scientific payload is a mission *requirement* carried
    # on the scenario, not a design variable on ``DesignVector``. The
    # mission's science team specifies payload mass and power; the
    # rover bus is then sized around it. Modelling payload here (rather
    # than folding it into ``chassis_mass_kg``) keeps the chassis input
    # purely structural, makes the bottom-up mass model reproduce
    # full-up published rover mass, and gives the optimiser the same
    # mass cost the real rover carried. See ``data/analytical/SCHEMA.md``
    # v9 entry for the full rationale.
    payload_mass_kg: float = Field(
        ge=0.0,
        le=30.0,
        default=0.0,
        description=(
            "Scientific-payload mass m_payload, kg, in [0, 30]. A mission "
            "requirement: the instrument suite the science team specifies "
            "(e.g. Pragyan APXS+LIBS ~3 kg, Yutu-2 GPR+VNIS+APXS ~25 kg). "
            "Added to total vehicle mass as a top-level line item "
            "*outside* the AIAA S-120A dry-mass growth margin "
            "(``m_total = m_dry + m_margin + m_payload``) because payload "
            "mass is a known requirement, not poorly-known bus hardware. "
            "Users can override via the /evaluate / /predict API "
            "parameter; the surrogate is trained on payload as an LHS "
            "feature (v9) so off-default queries keep calibrated PIs. "
            "Ceiling 30 kg covers the heaviest in-class lunar micro-rover "
            "payload (Yutu-2)."
        ),
    )
    payload_power_w: float = Field(
        ge=0.0,
        le=30.0,
        default=0.0,
        description=(
            "Scientific-payload continuous ops-time power draw P_payload, "
            "W, in [0, 30]. A mission requirement. Added to the "
            "continuous electrical load alongside avionics in the "
            "traverse power budget, and to the hot-case internal thermal "
            "dissipation. Users can override via the /evaluate / /predict "
            "API parameter; trained as an LHS feature (v9)."
        ),
    )


# ---------------------------------------------------------------------------
# Mission metrics (evaluator output)
# ---------------------------------------------------------------------------


class MissionMetrics(BaseModel):
    """Mission-level outputs of the evaluator or surrogate.

    All fields describe the **achievable performance** of a design under
    its scenario's effective duty cycle ``δ_eff = min(δ_des, δ_ops)``.
    Schema bumped to v6 (v6 schema update) when ``range_km`` was migrated from
    a tautological capability envelope to a real engineering metric
    responsive to drivetrain torque, soil, slope, and power budget.
    Real-rover-conservatism (Pragyan ~0.02, Yutu-2 ~0.015) is now
    expressed by lowering ``MissionScenario.operational_duty_cycle``
    rather than scaling the metric post-hoc.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Primary metrics - all evaluated at δ_eff
    range_km: float  # achievable distance over scenario duration at δ_eff
    energy_margin_pct: float  # SOC-based, clipped 0-100; reporting metric
    slope_capability_deg: float  # max climbable slope on this soil

    # Unclipped energy-balance signal for the surrogate. Defined as
    # ``(E_generated - E_consumed) / E_consumed * 100``, integrated over
    # the whole traverse. Unlike ``energy_margin_pct`` (SOC-based, clipped
    # at 0-100), this one is unbounded on both sides: negative means the
    # rover consumed more than it generated, >100 means surplus exceeded
    # consumption. Kept as a separate field so LHS-trained surrogates see
    # a smooth target; reporting gates keep using the clipped version.
    energy_margin_raw_pct: float = 0.0

    # Secondary metrics
    total_mass_kg: float
    peak_motor_torque_nm: float
    sinkage_max_m: float

    # Constraint flags
    thermal_survival: bool
    stalled: bool  # mirrors run_traverse(...).rover_stalled; replaces
    # the v5 ``motor_torque_ok`` field which was redundant with the
    # explicit torque-ceiling stall gate introduced in v6.

    # Optional uncertainty, populated by the surrogate layer
    range_km_std: float | None = None
    energy_margin_pct_std: float | None = None
    slope_capability_deg_std: float | None = None
