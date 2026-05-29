"""Pydantic v2 schemas exposed at the HTTP boundary.

Design goal: be a *thin* mirror of :mod:`roverdevkit.schema` so the
frontend can talk to the backend in terms of the same `DesignVector` /
`MissionScenario` objects the Python core uses. Where it makes sense,
we re-export the core models verbatim (frozen + extra-forbid is fine
over JSON); where the API value-add is non-trivial — `PredictRequest`,
`PredictResponse`, `RegistryEntrySummary`, etc. — we define a dedicated
boundary type so a future schema bump on the core does not silently
break the OpenAPI surface.

All response models have ``model_config = ConfigDict(frozen=True)`` so
they are safe to share across requests and so callers cannot mutate
cached registry / scenario payloads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from roverdevkit.schema import DesignVector, MissionScenario

# ---------------------------------------------------------------------------
# Shared mission-requirement override fields
# ---------------------------------------------------------------------------
#
# Schema v9: scientific payload is a *mission requirement* carried on
# ``MissionScenario`` (``payload_mass_kg`` / ``payload_power_w``), not a
# design-vector trade. Every request that resolves a scenario server-side
# accepts an optional per-call override so the Mission-Inputs panel (and
# the rediscovery harness) can substitute a mission's own payload without
# editing the canonical scenario catalogue. Both are LHS-sampled surrogate
# inputs over ``[0, 30]`` (mirroring the v7_1 δ_ops promotion), so any
# in-bounds override stays on the surrogate path with calibrated PIs.


def _payload_mass_field() -> Any:
    return Field(
        default=None,
        ge=0.0,
        le=30.0,
        description=(
            "Optional per-query override for "
            "``MissionScenario.payload_mass_kg`` (scientific-payload mass, "
            "kg, a mission requirement). ``None`` uses the scenario's "
            "class-typical default. Schema v9: payload mass is an "
            "LHS-sampled surrogate input over [0, 30], so any in-bounds "
            "override stays on the surrogate path with calibrated PIs."
        ),
    )


def _payload_power_field() -> Any:
    return Field(
        default=None,
        ge=0.0,
        le=30.0,
        description=(
            "Optional per-query override for "
            "``MissionScenario.payload_power_w`` (scientific-payload "
            "continuous ops-time power draw, W, a mission requirement). "
            "``None`` uses the scenario's class-typical default. Schema "
            "v9: LHS-sampled surrogate input over [0, 30]."
        ),
    )

# Re-export the core types unchanged. Pydantic v2 serialises both
# transparently to JSON; importing here keeps the OpenAPI schema names
# consistent with the Python core.
__all__ = [
    "DesignVector",
    "EvaluateMetric",
    "EvaluateRequest",
    "EvaluateResponse",
    "FeatureRow",
    "HealthResponse",
    "MissionScenario",
    "OptimizeCancelResponse",
    "OptimizeCheckpointOut",
    "OptimizeConstraintIn",
    "OptimizeJobResponse",
    "OptimizeObjectiveIn",
    "OptimizeParetoPoint",
    "OptimizeRequest",
    "OptimizeResultResponse",
    "PredictMode",
    "PredictRequest",
    "PredictResponse",
    "PredictTarget",
    "RediscoveryDetail",
    "RediscoveryListResponse",
    "RediscoverySummary",
    "RediscoveryParetoPoint",
    "RegistryEntrySummary",
    "RegistryListResponse",
    "ScenarioListResponse",
    "ScenarioWithSoil",
    "ShapExplainRequest",
    "ShapFeatureScore",
    "ShapLocalResponse",
    "SoilParametersOut",
    "StallDiagnosticOut",
    "SweepAxisIn",
    "SweepRequest",
    "SweepResponse",
    "SweepSensitivityOut",
    "ThermalDiagnosticOut",
    "VersionResponse",
]


# ---------------------------------------------------------------------------
# Health / version
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness + artifact-presence probe."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded"] = "ok"
    surrogate_loaded: bool
    surrogate_targets: list[str]
    quantile_bundles_path: str


class VersionResponse(BaseModel):
    """Static version metadata for the about box."""

    model_config = ConfigDict(frozen=True)

    api_version: str
    package_version: str
    dataset_version: str
    quantile_bundles_path: str


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class SoilParametersOut(BaseModel):
    """Bekker-Wong soil parameter snapshot, JSON-friendly.

    Mirrors :class:`roverdevkit.terramechanics.bekker_wong.SoilParameters`
    but as a plain Pydantic model so it serialises cleanly without
    dataclass field-ordering quirks.
    """

    model_config = ConfigDict(frozen=True)

    simulant: str
    n: float
    k_c: float
    k_phi: float
    cohesion_kpa: float
    friction_angle_deg: float
    shear_modulus_k_m: float


class ScenarioWithSoil(BaseModel):
    """Canonical mission scenario plus the nominal soil parameters.

    The soil block is included so the frontend can show the user what
    Bekker-Wong parameters were used as the surrogate's nominal soil
    values without an extra round-trip.
    """

    model_config = ConfigDict(frozen=True)

    scenario: MissionScenario
    soil: SoilParametersOut


class ScenarioListResponse(BaseModel):
    """List of canonical tradespace scenarios with brief metadata."""

    model_config = ConfigDict(frozen=True)

    scenarios: list[ScenarioWithSoil]


# ---------------------------------------------------------------------------
# Registry (real-rover validation set)
# ---------------------------------------------------------------------------


class RegistryEntrySummary(BaseModel):
    """A real-rover registry entry exposed to the frontend.

    Mirrors :class:`roverdevkit.validation.rover_registry.RoverRegistryEntry`
    excluding its non-JSON-friendly internals (the ``ThermalArchitecture``
    object). The thermal architecture is collapsed to a small dict so
    the frontend can show the user how the rover differs from the
    tradespace defaults without depending on the dataclass shape.
    """

    model_config = ConfigDict(frozen=True)

    rover_name: str
    is_flown: bool
    design: DesignVector
    scenario: MissionScenario
    gravity_m_per_s2: float
    thermal_architecture: dict[str, Any]
    panel_efficiency: float
    panel_dust_factor: float
    panel_tilt_deg: float
    panel_azimuth_deg: float
    imputation_notes: str


class RegistryListResponse(BaseModel):
    """All real-rover registry entries (flown and design-target tiers)."""

    model_config = ConfigDict(frozen=True)

    rovers: list[RegistryEntrySummary]


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


PrimaryTarget = Literal[
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
]


class FeatureRow(BaseModel):
    """The 27-D feature vector actually fed to the surrogate.

    Schema v9 added the two payload mission-requirement inputs
    (``scenario_payload_mass_kg`` / ``scenario_payload_power_w``),
    taking the surrogate input frame from 25 to 27 columns.

    Echoed back so the frontend can show the nominal soil / categorical
    values that were used; useful for "did I really pick the soil I
    thought I picked?" sanity checks and as the basis for OOD warnings
    in later steps.
    """

    model_config = ConfigDict(frozen=True)

    columns: list[str]
    values: list[Any]


class PredictRequest(BaseModel):
    """Input payload for :http:post:`/predict`.

    The user always submits a full :class:`DesignVector` (the schema's
    own bounds validation will reject anything outside the design
    space) plus a canonical scenario name. The scenario's nominal soil
    parameters are looked up server-side from the soil catalogue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: DesignVector
    scenario_name: str = Field(
        description="Canonical scenario key (one of the four returned by /scenarios)."
    )
    operational_duty_cycle: float | None = Field(
        default=None,
        ge=0.0,
        le=0.6,
        description=(
            "Optional per-query override for "
            "``MissionScenario.operational_duty_cycle``. SCHEMA_VERSION "
            "v7_1 (v7_1 schema follow-on): δ_ops is now a per-row LHS "
            "feature uniform on [0, 0.6], so any in-bounds override "
            "stays on the surrogate path with calibrated PIs. The "
            "pre-v7_1 evaluator-only fallback for off-default values "
            "has been removed."
        ),
    )
    payload_mass_kg: float | None = _payload_mass_field()
    payload_power_w: float | None = _payload_power_field()
    repair_crossings: bool = Field(
        default=True,
        description=(
            "Row-wise sort the (q05, q50, q95) triple before returning. "
            "Cheap, never worsens empirical coverage, and avoids "
            "non-monotone reports to the frontend. Set False to inspect "
            "raw model output."
        ),
    )


class PredictTarget(BaseModel):
    """Per-target prediction triple."""

    model_config = ConfigDict(frozen=True)

    target: PrimaryTarget
    q05: float
    q50: float
    q95: float


PredictMode = Literal["surrogate", "evaluator_only"]
"""Kept as a literal for response-schema stability across the v6 ->
v7_1 transition. Live ``/predict`` always returns ``"surrogate"``
since v7_1; the ``"evaluator_only"`` slot is retained for forwards-
compat with future evaluator-fallback paths (e.g. out-of-bounds
inputs the surrogate refuses to predict on)."""


class PredictResponse(BaseModel):
    """Median + 90 % PI for each primary regression target.

    See ``reports/intervals_v4/SUMMARY.md`` for empirical coverage
    on the test split (target ≈ 90 %, achieved 86–92 % per scenario).

    SCHEMA_VERSION v7_1 (v7_1 schema follow-on): ``operational_duty_cycle``
    is a true surrogate input feature, so any in-bounds δ_ops stays on
    the surrogate path. ``mode`` is therefore always ``"surrogate"`` in
    v7_1; the literal still admits ``"evaluator_only"`` for forwards-
    compat with future fallback paths.
    """

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    predictions: list[PredictTarget]
    feature_row: FeatureRow
    mode: PredictMode = "surrogate"
    """Always ``"surrogate"`` in v7_1; reserved literal slot for future
    evaluator fallbacks. See :class:`PredictRequest` for the override
    semantics."""


# ---------------------------------------------------------------------------
# Evaluate (deterministic mission evaluator with SCM correction)
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    """Input payload for :http:post:`/evaluate`.

    Drives the corrected mission evaluator
    (:func:`roverdevkit.mission.evaluator.evaluate`, BW + wheel-level
    SCM correction) on a single ``DesignVector`` and a canonical
    scenario. Used by the single-design panel as the source of truth
    for the median value of each performance metric; the surrogate's
    quantile heads supply the prediction-interval band around it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: DesignVector
    scenario_name: str = Field(
        description="Canonical scenario key (one of the four returned by /scenarios)."
    )
    operational_duty_cycle: float | None = Field(
        default=None,
        ge=0.0,
        le=0.6,
        description=(
            "Optional per-query override for "
            "``MissionScenario.operational_duty_cycle``. Schema v7: the "
            "evaluator uses this value directly as δ_eff (clamped to "
            "[0, 1]). ``None`` uses the scenario's calibrated default."
        ),
    )
    payload_mass_kg: float | None = _payload_mass_field()
    payload_power_w: float | None = _payload_power_field()


class EvaluateMetric(BaseModel):
    """Per-target deterministic value from the corrected evaluator."""

    model_config = ConfigDict(frozen=True)

    target: PrimaryTarget
    value: float


class ThermalDiagnosticOut(BaseModel):
    """Per-design output of the lumped-parameter thermal model.

    The single-design panel surfaces both temperatures so users can
    see *why* a thermal-survival flag fired (it's almost always the
    cold case for micro-rovers without RHUs). ``rhu_power_w`` is
    included because it's the most common knob users would reach for
    if they were sizing a real rover; in our design vector it is
    fixed at 0 W by convention -- thermal is a diagnostic, not a
    design lever, since baseline-surrogate.
    """

    model_config = ConfigDict(frozen=True)

    survives: bool
    """End-to-end pass / fail (= ``hot_case_ok and cold_case_ok``)."""

    peak_sun_temp_c: float
    lunar_night_temp_c: float
    min_operating_temp_c: float
    max_operating_temp_c: float
    rhu_power_w: float
    hibernation_power_w: float
    surface_area_m2: float
    hot_case_ok: bool
    cold_case_ok: bool


class StallDiagnosticOut(BaseModel):
    """Drivetrain stall status and the torque numbers that drove it.

    SCHEMA_VERSION v6 (v6 schema update): replaces ``MotorTorqueDiagnosticOut``.
    The pre-v6 diagnostic flagged ``motor_torque_ok`` whenever the
    per-step peak torque stayed below an implicit, mass-derived ceiling
    inside the mass model. v6 makes the ceiling explicit
    (``DesignVector.peak_wheel_torque_nm``) and surfaces the stall gate
    directly. ``stalled = True`` means the slip-balance torque demand
    exceeded the design's drivetrain capacity *or* the slip solver
    couldn't develop the required drawbar pull, equivalent to
    ``MissionMetrics.stalled`` and the underlying
    ``run_traverse(...).rover_stalled`` flag.
    """

    model_config = ConfigDict(frozen=True)

    stalled: bool
    """``True`` iff the rover's drivetrain stalled on the scenario's
    worst-case slope. Replaces the v5 ``survives`` field."""

    peak_torque_demand_nm: float
    """Per-wheel hub torque the slip-balance solve demanded."""

    peak_torque_capacity_nm: float
    """``DesignVector.peak_wheel_torque_nm`` echoed back for context."""


class EvaluateResponse(BaseModel):
    """Deterministic evaluator output for the four primary regression targets.

    Values match :class:`roverdevkit.schema.MissionMetrics` 1:1 for the
    primary subset; the response also surfaces structured constraint
    diagnostics (``thermal``, ``stall``) so the frontend can explain *why*
    a flag fired without a second round-trip.
    """

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    metrics: list[EvaluateMetric]
    thermal: ThermalDiagnosticOut
    stall: StallDiagnosticOut
    """Schema v6: replaces the v5 ``motor_torque`` field."""
    effective_duty_cycle: float
    """Schema v7: ``operational_duty_cycle`` (per-scenario default or
    per-call override) clamped to ``[0, 1]``. The v6 ``min(δ_des,
    δ_ops)`` semantics collapsed when ``designed_duty_cycle`` was
    removed from the design vector. Surfaced so the single-design
    panel can echo the duty the evaluator actually drove the rover at."""
    cruise_speed_mps: float
    """Derived rover cruise speed used by the time loop. Replaces the
    v5 ``DesignVector.nominal_speed_mps`` design input. See
    :func:`roverdevkit.drivetrain.motor.cruise_speed`."""
    used_scm_correction: bool
    elapsed_ms: float


class SweepAxisIn(BaseModel):
    """One axis of a parametric sweep (mirror of ``SweepAxis``).

    The variable name is validated server-side against
    :data:`roverdevkit.tradespace.sweeps.SWEEPABLE_VARIABLES`; the
    ``lo`` / ``hi`` range is validated against the ``DesignVector``
    schema bounds inside :func:`roverdevkit.tradespace.sweeps.expand_grid`
    (Pydantic on the per-cell ``DesignVector`` rebuild surfaces the
    out-of-bounds case as a ValidationError -> 422).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variable: str
    lo: float
    hi: float
    n_points: int = Field(ge=2, le=200)


class SweepRequest(BaseModel):
    """``POST /sweep`` body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    """One of the primary regression targets (range_km,
    energy_margin_raw_pct, slope_capability_deg, total_mass_kg)."""

    x_axis: SweepAxisIn
    y_axis: SweepAxisIn | None = None

    base_design: DesignVector
    """The "rest of the design": every dimension not on an axis is
    held at this value across the whole grid."""

    scenario_name: str
    backend: Literal["auto", "evaluator", "surrogate"] = "auto"
    operational_duty_cycle: float | None = Field(
        default=None,
        ge=0.0,
        le=0.6,
        description=(
            "Optional per-query override for "
            "``MissionScenario.operational_duty_cycle``. SCHEMA_VERSION "
            "v7_1: δ_ops is a true LHS-sampled surrogate input, so any "
            "in-bounds override stays on the surrogate sweep path with "
            "calibrated quantiles; the deterministic-evaluator sweep "
            "path also honours it (one δ_ops per grid, the grid still "
            "runs one-shot)."
        ),
    )
    payload_mass_kg: float | None = _payload_mass_field()
    payload_power_w: float | None = _payload_power_field()


class SweepSensitivityOut(BaseModel):
    """Mirror of :class:`roverdevkit.tradespace.sweeps.SweepSensitivity`.

    All values share the units of the swept target metric. ``relative_spread``
    is dimensionless: the absolute spread divided by the larger of
    ``|max|``, ``|min|``, ε. Frontend uses it to decide whether the metric
    is effectively flat across the grid.
    """

    model_config = ConfigDict(frozen=True)

    total_spread: float
    relative_spread: float
    axis_spread_x: float
    axis_spread_y: float | None


class SweepResponse(BaseModel):
    """Sweep grid + the values matrix + provenance.

    ``z_values`` is row-major: 1-D ``(n_x,)`` for a 1-D sweep,
    2-D ``(n_y, n_x)`` for 2-D. The 2-D shape matches Plotly's
    heatmap convention (rows = y, columns = x) so the frontend can
    pass it through unchanged.
    """

    model_config = ConfigDict(frozen=True)

    target: str
    scenario_name: str
    x_variable: str
    y_variable: str | None
    x_values: list[float]
    y_values: list[float] | None
    z_values: list[float] | list[list[float]]
    backend_used: Literal["evaluator", "surrogate"]
    backend_requested: Literal["auto", "evaluator", "surrogate"]
    used_scm_correction: bool
    """``True`` when the evaluator path ran with the SCM correction
    artifact loaded; ``False`` when running on BW-only fallback or
    when the surrogate path served the request (the surrogate was
    already trained on a corrected corpus)."""

    n_cells: int
    elapsed_ms: float

    sensitivity: SweepSensitivityOut
    """Per-axis spread of the swept metric. Drives the inline sensitivity
    hint under the chart so users can tell at a glance when a metric is
    saturated on the chosen grid or when one axis dominates the other."""


# ---------------------------------------------------------------------------
# Optimize (NSGA-II job orchestration)
# ---------------------------------------------------------------------------


class OptimizeObjectiveIn(BaseModel):
    """One Pareto objective requested by the UI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: PrimaryTarget
    direction: Literal["min", "max"]


class OptimizeConstraintIn(BaseModel):
    """Threshold constraint over a primary target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: PrimaryTarget
    sense: Literal["min", "max"]
    value: float


class OptimizeRequest(BaseModel):
    """``POST /optimize`` body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_name: str = Field(
        description="Canonical scenario key (one of the four returned by /scenarios)."
    )
    backend: Literal["surrogate", "evaluator"] = Field(
        default="evaluator",
        description=(
            "Corrected physics evaluator by default; capped server-side at "
            "5000 evaluations so a live job finishes inside ~2 min. The "
            "surrogate backend is accepted as an opt-in benchmarking option."
        ),
    )
    objectives: list[OptimizeObjectiveIn] = Field(
        default_factory=lambda: [
            OptimizeObjectiveIn(target="range_km", direction="max"),
            OptimizeObjectiveIn(target="total_mass_kg", direction="min"),
            OptimizeObjectiveIn(target="slope_capability_deg", direction="max"),
        ],
        min_length=1,
        max_length=4,
    )
    constraints: list[OptimizeConstraintIn] = Field(default_factory=list, max_length=8)
    population_size: int = Field(default=64, ge=4, le=300)
    n_generations: int = Field(default=100, ge=1, le=500)
    seed: int = Field(default=0, ge=0)
    operational_duty_cycle: float | None = Field(
        default=None,
        ge=0.0,
        le=0.6,
        description="Optional per-job override for MissionScenario.operational_duty_cycle.",
    )
    payload_mass_kg: float | None = _payload_mass_field()
    payload_power_w: float | None = _payload_power_field()


class OptimizeJobResponse(BaseModel):
    """Immediate response after queueing an optimization job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "completed", "cancelled", "failed"]
    stream_url: str
    result_url: str
    cancel_url: str


class OptimizeCheckpointOut(BaseModel):
    """Per-generation SSE payload."""

    model_config = ConfigDict(frozen=True)

    gen: int
    hypervolume: float
    pareto_size: int
    best_per_objective: dict[str, float]


class OptimizeParetoPoint(BaseModel):
    """One final Pareto-front point."""

    model_config = ConfigDict(frozen=True)

    design: DesignVector
    metrics: dict[str, float]


class OptimizeResultResponse(BaseModel):
    """Final job state and Pareto front."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "completed", "cancelled", "failed"]
    backend_used: Literal["surrogate", "evaluator"] | None = None
    checkpoints: list[OptimizeCheckpointOut] = Field(default_factory=list)
    pareto_front: list[OptimizeParetoPoint] = Field(default_factory=list)
    error: str | None = None


class OptimizeCancelResponse(BaseModel):
    """Response from ``POST /optimize/{id}/cancel``."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: Literal["queued", "running", "completed", "cancelled", "failed"]


class ShapFeatureScore(BaseModel):
    """Per-feature contribution to a single-design prediction."""

    model_config = ConfigDict(frozen=True)

    feature: str
    value: float


class ShapExplainRequest(BaseModel):
    """Explain the current design for one target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    design: DesignVector
    scenario_name: str
    target: PrimaryTarget
    operational_duty_cycle: float | None = Field(default=None, ge=0.0, le=0.6)
    payload_mass_kg: float | None = _payload_mass_field()
    payload_power_w: float | None = _payload_power_field()


class ShapLocalResponse(BaseModel):
    """Per-design feature contributions for one target prediction."""

    model_config = ConfigDict(frozen=True)

    target: PrimaryTarget
    prediction: float
    base_value: float
    contributions: list[ShapFeatureScore]


# ---------------------------------------------------------------------------
# Rediscovery validation (Layer-5 LOO sweep, paper headline figure)
# ---------------------------------------------------------------------------


RediscoveryBackend = Literal["evaluator", "surrogate"]
"""Which fitness backend produced the precomputed rediscovery artifacts.

The evaluator-backed sweep
(``reports/rediscovery_loo_evaluator/``) is the source of truth for
the paper figure: every front point is corrected-evaluator-truth.
The surrogate-backed sweep (``reports/rediscovery_loo_surrogate_v8/``)
is offered as a faster reference alongside it; per the 2026-05-28
panel-tilt fix in ``reports/rediscovery_loo_comparison.md``, the
v8 surrogate's polar-front predictions are still horizontal-panel
based, so it should be read as a wall-clock benchmark, not a
fidelity check at high latitudes."""


class RediscoveryParetoPoint(BaseModel):
    """One point on the rediscovery Pareto front.

    The metrics block stores the four primary regression targets
    evaluated for this design under the rover's class-generic
    scenario (``polar_micro``, ``mare_micro``, etc.). Not every
    rediscovery JSON carries every target on every point — older
    artifacts predate the schema-v6 stall flag — so the metrics
    dict is left open-shape rather than a fixed PrimaryTarget map.
    """

    model_config = ConfigDict(frozen=True)

    design: DesignVector
    metrics: dict[str, float]


class RediscoverySummary(BaseModel):
    """One rover's rediscovery result, summary view.

    Mirrors the per-rover row of the published Layer-5 report
    (``reports/rediscovery_loo_evaluator/rediscovery_loo_report.md``)
    plus the registry's ``is_flown`` tier so the frontend can label
    flown rovers (Pragyan, Yutu-2) separately from design-target
    registry cases (MoonRanger, Rashid-1, CADRE-unit, Tenacious).
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    """URL-safe identifier matching the per-rover JSON file stem
    in ``reports/rediscovery_loo_evaluator/``. Use as the path
    component on the detail endpoint."""

    rover_name: str
    """Human-readable name from the registry (e.g. ``"CADRE-unit"``)."""

    is_flown: bool
    """``True`` for rovers that have actually flown a lunar
    mission (Pragyan, Yutu-2). Other registry rovers are
    well-spec'd design-target lunar micro-rovers added for
    Layer-5 OOD coverage."""

    class_generic_scenario: str
    """Class-generic ``*_micro`` scenario the rediscovery harness
    placed this rover under (e.g. ``"polar_micro"``)."""

    backend: RediscoveryBackend
    design_space_distance: float
    pareto_dominated: bool
    mass_budget_kg: float
    pareto_front_size: int


class RediscoveryListResponse(BaseModel):
    """All rovers' rediscovery summaries for a single backend."""

    model_config = ConfigDict(frozen=True)

    backend: RediscoveryBackend
    rovers: list[RediscoverySummary]


class RediscoveryDetail(BaseModel):
    """Full rediscovery result for one rover under one backend.

    Includes the summary fields plus the Pareto front itself, the
    rover's own metrics under the class-generic scenario, the
    per-variable percent error against the nearest Pareto point,
    and the integer-match flags for ``grouser_count`` /
    ``n_wheels``.
    """

    model_config = ConfigDict(frozen=True)

    slug: str
    rover_name: str
    is_flown: bool
    class_generic_scenario: str
    backend: RediscoveryBackend

    rover_design: DesignVector
    """The real rover's published design vector. Plotted as the
    overlay marker on the Pareto-front figure."""

    rover_metrics_under_generic_scenario: dict[str, float]
    """Metrics from re-evaluating ``rover_design`` under the
    class-generic scenario (with the 2026-05-28 panel-tilt fix
    applied — see ``reports/rediscovery_loo_comparison.md``)."""

    design_space_distance: float
    pareto_dominated: bool
    mass_budget_kg: float
    integer_matches: dict[str, bool]
    per_variable_percent_errors: dict[str, float]

    nearest_pareto_index: int
    nearest_pareto_design: DesignVector
    nearest_pareto_metrics: dict[str, float]

    pareto_front: list[RediscoveryParetoPoint]
