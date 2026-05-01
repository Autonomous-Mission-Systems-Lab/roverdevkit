"""Shared data schemas for design vectors, scenarios, and mission metrics.

These are the canonical types that flow between the mission evaluator,
surrogate, and tradespace layers. Using Pydantic gives us validation at the
boundaries (e.g. reject a wheel radius outside the design-space bounds) and
free JSON/YAML serialization for scenario config files.

Design-variable ranges follow :file:`project_plan.md` §3.1.
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

    Schema v7 (W12 step B follow-up) consolidated the v6 designed /
    operational duty-cycle split back into a single per-scenario
    ``operational_duty_cycle``. The v6 ``designed_duty_cycle`` design
    field carried no engineering content — the v6 mass model never
    actually scaled with it (battery, thermal, and motor masses size
    on capacity / area / torque-capacity respectively, none on duty
    cycle) — so the only role of ``δ_des`` was to act as an upper
    bound on ``δ_eff``, which a user can equivalently express by
    lowering ``operational_duty_cycle``. Removing it gets us a clean
    11-D design space, matches surrogate dimensionality to the
    user-facing form, and removes the LHS-saturation artefact in
    high-δ_des / low-δ_ops cells. See
    ``reports/week12_design/decision.md`` for the full rationale.
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
            "dataset-regen task tracked in the project log."
        ),
    )
    grouser_count: int = Field(ge=0, le=24, description="Number of grousers N_g")
    n_wheels: Literal[4, 6] = Field(description="Wheel count N_w")

    # Chassis
    chassis_mass_kg: float = Field(
        ge=3.0,
        le=50.0,
        description=(
            "Dry chassis mass m_c. Upper bound 50 kg widened from 35 kg "
            "in the v3 LHS bounds widening so the heavier flown lunar "
            "micro-rovers (Yutu-class, ~30-40 kg ex-payload) sit inside "
            "the surrogate's training support rather than at a corner. "
            "Floor 3 kg keeps the design space anchored to actual "
            "micro-rover scale."
        ),
    )
    wheelbase_m: float = Field(ge=0.3, le=1.2, description="Wheelbase L_wb")

    # Power
    solar_area_m2: float = Field(ge=0.1, le=1.5, description="Solar array area A_s")
    battery_capacity_wh: float = Field(ge=20.0, le=500.0, description="Battery capacity C_b")
    avionics_power_w: float = Field(ge=5.0, le=40.0, description="Continuous avionics draw P_a")

    # Operations
    peak_wheel_torque_nm: float = Field(
        ge=0.3,
        le=20.0,
        description=(
            "Peak per-wheel hub torque T_hub^peak that the drivetrain "
            "(motor + gearbox combined) can sustain. Sizes motor mass via "
            "the parametric mass model and gates whether the rover stalls "
            "on slope. Cruise speed is *derived* from this, the slip-balance "
            "torque demand, and the steady-state power budget — not a free "
            "design variable. Bounds: 0.3 Nm captures the smallest published "
            "lunar micro-rover (Rashid-1, ~0.3 Nm peak); 20 Nm covers "
            "over-sized direct-drive concepts at the top of the design "
            "space. Schema-bumped to v6 (W12 step B) to replace "
            "``nominal_speed_mps`` with a true drivetrain capability "
            "input; see reports/week12_design/decision.md."
        ),
    )
    # Schema v7 (W12 step B follow-up) removed the v6
    # ``designed_duty_cycle`` field. Drive duty cycle is now a
    # per-scenario quantity (``MissionScenario.operational_duty_cycle``)
    # with an optional per-call API override; see the class docstring
    # above for the rationale.


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
    Phase 3; validation scenarios (Week 5 rover-comparison harness)
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
            "Default ground-ops drive duty for this scenario, in [0, 0.6]. "
            "The evaluator uses this directly as δ_eff (clamped to "
            "[0, 1]); schema v7 removed the v6 ``designed_duty_cycle`` "
            "design field after that field turned out to do no engineering "
            "work in the v6 mass model. Calibrated against published "
            "rover-on-mission ops cadence (mare 0.30, crater 0.20, highland "
            "0.15, polar 0.05; see reports/week12_design/decision.md). "
            "Users can override via the /evaluate / /predict API "
            "parameter; the surrogate is only trained at scenario defaults "
            "so off-default queries fall back to the deterministic "
            "evaluator (no PIs)."
        ),
    )


# ---------------------------------------------------------------------------
# Mission metrics (evaluator output)
# ---------------------------------------------------------------------------


class MissionMetrics(BaseModel):
    """Mission-level outputs of the evaluator or surrogate.

    All fields describe the **achievable performance** of a design under
    its scenario's effective duty cycle ``δ_eff = min(δ_des, δ_ops)``.
    Schema bumped to v6 (W12 step B) when ``range_km`` was migrated from
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
    # consumption. Kept as a separate field so Week-6 LHS surrogates see
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
