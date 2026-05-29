/**
 * TypeScript mirrors of the FastAPI Pydantic schemas.
 *
 * These are deliberately hand-written rather than generated from the
 * OpenAPI doc: the public surface is small, the manual definitions
 * give us better doc comments at the call sites, and they double as
 * the source of truth for design-vector bounds in the form UI. If
 * the API surface grows past ~10 routes we should switch to
 * `openapi-typescript` codegen as a build step.
 */

/** Lunar mission scenarios the surrogate is calibrated for. */
export type ScenarioName =
  | "equatorial_mare_traverse"
  | "polar_prospecting"
  | "highland_slope_capability"
  | "crater_rim_survey";

/** Subset of `MissionScenario.terrain_class` exposed via the API. */
export type TerrainClass =
  | "mare_nominal"
  | "mare_loose"
  | "highland_dense"
  | "polar_regolith";

export type SunGeometry = "continuous" | "diurnal" | "polar_intermittent";

/**
 * Mirror of `roverdevkit.schema.DesignVector` (schema v7, v6 schema update
 * follow-up).
 *
 * v6 changes: `nominal_speed_mps` removed (cruise speed is derived in
 * the evaluator from drivetrain torque + slip-balance + energy-balance
 * + kinematic envelope); `peak_wheel_torque_nm` added as a true
 * drivetrain-capability input.
 *
 * v7 changes: `designed_duty_cycle` removed after that field turned
 * out to do no engineering work in the v6 mass model. Drive duty
 * cycle now lives entirely on the scenario
 * (`MissionScenario.operational_duty_cycle`) with optional per-call
 * override at inference time.
 */
export interface DesignVector {
  wheel_radius_m: number;
  wheel_width_m: number;
  grouser_height_m: number;
  grouser_count: number;
  n_wheels: 4 | 6;
  chassis_mass_kg: number;
  wheelbase_m: number;
  solar_area_m2: number;
  battery_capacity_wh: number;
  avionics_power_w: number;
  peak_wheel_torque_nm: number;
}

/**
 * Mirror of `roverdevkit.schema.MissionScenario`.
 *
 * `operational_duty_cycle` is the per-scenario ground-ops drive duty
 * and, since schema v7, the *only* drive duty parameter. The
 * evaluator uses it directly as δ_eff (clamped to [0, 1]). The
 * frontend reads the calibrated default here and lets the user
 * override it via the "Operations" panel.
 */
export interface MissionScenario {
  name: string;
  latitude_deg: number;
  traverse_distance_m: number;
  terrain_class: TerrainClass;
  soil_simulant: string;
  mission_duration_earth_days: number;
  max_slope_deg: number;
  sun_geometry: SunGeometry;
  operational_duty_cycle: number;
  /**
   * Scientific-payload mass (kg) and continuous ops-time power (W),
   * mission requirements introduced in schema v9. Payload mass is added
   * to total vehicle mass as a top-level line item *outside* the dry-mass
   * growth margin; payload power adds to the continuous electrical load.
   * Each scenario ships a class-typical default; the Mission Inputs panel
   * lets a user override both for a single round-trip.
   */
  payload_mass_kg: number;
  payload_power_w: number;
}

export interface SoilParametersOut {
  simulant: string;
  n: number;
  k_c: number;
  k_phi: number;
  cohesion_kpa: number;
  friction_angle_deg: number;
  shear_modulus_k_m: number;
}

export interface ScenarioWithSoil {
  scenario: MissionScenario;
  soil: SoilParametersOut;
}

export interface ScenarioListResponse {
  scenarios: ScenarioWithSoil[];
}

export type PrimaryTarget =
  | "range_km"
  | "energy_margin_raw_pct"
  | "slope_capability_deg"
  | "total_mass_kg";

/**
 * Canonical row order used everywhere the four primary targets are
 * rendered. Matches `roverdevkit.surrogate.features.PRIMARY_REGRESSION_TARGETS`
 * so the Python and TypeScript layers agree by construction.
 */
export const PRIMARY_REGRESSION_TARGET_ORDER: readonly PrimaryTarget[] = [
  "range_km",
  "energy_margin_raw_pct",
  "slope_capability_deg",
  "total_mass_kg",
] as const;

export interface PredictTarget {
  target: PrimaryTarget;
  q05: number;
  q50: number;
  q95: number;
}

export interface FeatureRow {
  columns: string[];
  values: unknown[];
}

export interface PredictRequest {
  design: DesignVector;
  scenario_name: string;
  /**
   * Optional per-query override for `MissionScenario.operational_duty_cycle`.
   * SCHEMA_VERSION v7_1: δ_ops is a true LHS-sampled surrogate input,
   * so any in-bounds override stays on the surrogate path with
   * calibrated PIs (`mode = "surrogate"`).
   */
  operational_duty_cycle?: number | null;
  /**
   * Optional per-query overrides for the schema-v9 payload mission
   * requirements (`MissionScenario.payload_mass_kg` / `payload_power_w`).
   * `null`/omitted uses the scenario's class-typical default. Both are
   * LHS-sampled surrogate inputs over [0, 30], so any in-bounds override
   * stays on the surrogate path with calibrated PIs.
   */
  payload_mass_kg?: number | null;
  payload_power_w?: number | null;
  repair_crossings?: boolean;
}

/**
 * Mirror of the FastAPI `PredictMode`. SCHEMA_VERSION v7_1 always
 * returns `"surrogate"`; the `"evaluator_only"` literal is retained
 * for forwards-compat with any future evaluator-fallback paths (e.g.
 * out-of-bounds inputs). The frontend keeps the `<NoPiBanner />`
 * gate on `mode` so a future fallback degrades cleanly without a UI
 * change.
 */
export type PredictMode = "surrogate" | "evaluator_only";

export interface PredictResponse {
  scenario_name: string;
  quantiles: [number, number, number];
  predictions: PredictTarget[];
  feature_row: FeatureRow;
  mode: PredictMode;
}

/**
 * Mirror of the FastAPI `EvaluateRequest`. Drives the deterministic
 * corrected mission evaluator on a single design × canonical scenario.
 * Used by the single-design panel as the source of truth for the
 * median value of each performance metric; the surrogate's quantile
 * heads supply the prediction-interval band around it.
 */
export interface EvaluateRequest {
  design: DesignVector;
  scenario_name: string;
  /**
   * Optional per-query override for `MissionScenario.operational_duty_cycle`.
   * Schema v7: the evaluator uses this value directly as δ_eff
   * (clamped to [0, 1]); when omitted we use the scenario's
   * calibrated default.
   */
  operational_duty_cycle?: number | null;
  /**
   * Optional per-query overrides for the schema-v9 payload mission
   * requirements (`MissionScenario.payload_mass_kg` / `payload_power_w`).
   * `null`/omitted uses the scenario's class-typical default.
   */
  payload_mass_kg?: number | null;
  payload_power_w?: number | null;
}

export interface EvaluateMetric {
  target: PrimaryTarget;
  value: number;
}

/**
 * Mirror of the FastAPI `ThermalDiagnosticOut`. Every numeric field is
 * already in the user's display units (°C, W, m²) so the panel and
 * dialog can render them without conversion.
 */
export interface ThermalDiagnostic {
  survives: boolean;
  peak_sun_temp_c: number;
  lunar_night_temp_c: number;
  min_operating_temp_c: number;
  max_operating_temp_c: number;
  rhu_power_w: number;
  hibernation_power_w: number;
  surface_area_m2: number;
  hot_case_ok: boolean;
  cold_case_ok: boolean;
}

/**
 * Mirror of the FastAPI `StallDiagnosticOut` (schema v6).
 *
 * Replaces the v5 `MotorTorqueDiagnostic`. The drivetrain stalls when
 * the slip-balance torque demand exceeds the design's
 * `peak_wheel_torque_nm` capacity, or when the slip solver cannot
 * develop the required drawbar pull on the scenario's worst-case slope.
 */
export interface StallDiagnostic {
  stalled: boolean;
  peak_torque_demand_nm: number;
  peak_torque_capacity_nm: number;
}

export interface EvaluateResponse {
  scenario_name: string;
  metrics: EvaluateMetric[];
  thermal: ThermalDiagnostic;
  /** Schema v6: replaces the v5 `motor_torque` field. */
  stall: StallDiagnostic;
  /**
   * Schema v7: `operational_duty_cycle` (per-scenario default or
   * per-call override) clamped to [0, 1]. The v6 `min(δ_des, δ_ops)`
   * semantics collapsed when `designed_duty_cycle` was removed from
   * the design vector. Surfaced so the single-design panel can echo
   * the duty the evaluator actually drove the rover at.
   */
  effective_duty_cycle: number;
  /**
   * Derived rover cruise speed used by the time loop. Replaces the v5
   * `DesignVector.nominal_speed_mps` design input.
   */
  cruise_speed_mps: number;
  used_scm_correction: boolean;
  elapsed_ms: number;
}

/**
 * Merged per-target row consumed by the chart and the panel table.
 *
 * - `value` is the deterministic median from the evaluator (ground truth).
 * - `q05`/`q95` are the surrogate's calibrated 90% prediction interval
 *   wrapping that median. Both may be `undefined` while the
 *   corresponding request is in flight or has failed.
 */
export interface PredictionRow {
  target: PrimaryTarget;
  value: number;
  q05: number | null;
  q95: number | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  surrogate_loaded: boolean;
  surrogate_targets: string[];
  quantile_bundles_path: string;
}

export interface VersionResponse {
  api_version: string;
  package_version: string;
  dataset_version: string;
  quantile_bundles_path: string;
}

/**
 * Mirror of the FastAPI `SweepAxisIn` schema. A sweep axis defines a
 * linearly-spaced grid `[lo, hi]` over a single design-vector field
 * with `n_points` cells (inclusive at both ends).
 */
export interface SweepAxisIn {
  variable: SweepableVariable;
  lo: number;
  hi: number;
  n_points: number;
}

/**
 * Subset of `DesignVector` keys the sweep page lets the user vary on
 * a grid axis. Mirrors `roverdevkit.tradespace.sweeps.SWEEPABLE_VARIABLES`;
 * `n_wheels` is excluded because it is binary.
 */
export type SweepableVariable =
  | "wheel_radius_m"
  | "wheel_width_m"
  | "grouser_height_m"
  | "grouser_count"
  | "chassis_mass_kg"
  | "wheelbase_m"
  | "solar_area_m2"
  | "battery_capacity_wh"
  | "avionics_power_w"
  | "peak_wheel_torque_nm";

export const SWEEPABLE_VARIABLES: readonly SweepableVariable[] = [
  "wheel_radius_m",
  "wheel_width_m",
  "grouser_height_m",
  "grouser_count",
  "chassis_mass_kg",
  "wheelbase_m",
  "solar_area_m2",
  "battery_capacity_wh",
  "avionics_power_w",
  "peak_wheel_torque_nm",
] as const;

export type SweepBackend = "auto" | "evaluator" | "surrogate";

export interface SweepRequest {
  target: PrimaryTarget;
  x_axis: SweepAxisIn;
  y_axis?: SweepAxisIn | null;
  base_design: DesignVector;
  scenario_name: string;
  backend?: SweepBackend;
  /**
   * Optional per-query override for `MissionScenario.operational_duty_cycle`.
   * SCHEMA_VERSION v7_1: δ_ops is a true LHS-sampled surrogate input,
   * so the override is honoured on both sweep backends — surrogate
   * batch predict and per-cell deterministic evaluator — keeping the
   * sweep tab in sync with the Single design tab's slider.
   */
  operational_duty_cycle?: number | null;
  /**
   * Optional per-query overrides for the schema-v9 payload mission
   * requirements. Held constant across the grid; only design variables
   * vary on the sweep axes.
   */
  payload_mass_kg?: number | null;
  payload_power_w?: number | null;
}

export interface SweepResponse {
  target: PrimaryTarget;
  scenario_name: string;
  x_variable: SweepableVariable;
  y_variable: SweepableVariable | null;
  x_values: number[];
  y_values: number[] | null;
  /** 1-D `(n_x,)` for a 1-D sweep, 2-D `(n_y, n_x)` for a 2-D sweep. */
  z_values: number[] | number[][];
  backend_used: "evaluator" | "surrogate";
  backend_requested: SweepBackend;
  used_scm_correction: boolean;
  n_cells: number;
  elapsed_ms: number;
  sensitivity: SweepSensitivity;
}

/**
 * Per-axis spread of the swept metric. Powers the inline sensitivity hint
 * shown under the chart so the user can quickly tell when a metric is
 * effectively flat across the chosen grid (saturation), or when one axis
 * dominates the other by an order of magnitude (visual masking).
 */
export interface SweepSensitivity {
  /** max(z) - min(z) over the whole grid, in target units. */
  total_spread: number;
  /** total_spread / max(|max|, |min|, eps); dimensionless. */
  relative_spread: number;
  /** Median marginal x-spread (1-D = total_spread). */
  axis_spread_x: number;
  /** Median marginal y-spread; null for 1-D sweeps. */
  axis_spread_y: number | null;
}

export type OptimizeBackend = "surrogate" | "evaluator";
export type ObjectiveDirection = "min" | "max";
export type ConstraintSense = "min" | "max";
export type OptimizeJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "cancelled"
  | "failed";

export interface OptimizeObjectiveIn {
  target: PrimaryTarget;
  direction: ObjectiveDirection;
}

export interface OptimizeConstraintIn {
  target: PrimaryTarget;
  sense: ConstraintSense;
  value: number;
}

export interface OptimizeRequest {
  scenario_name: string;
  backend?: OptimizeBackend;
  objectives: OptimizeObjectiveIn[];
  constraints?: OptimizeConstraintIn[];
  population_size?: number;
  n_generations?: number;
  seed?: number;
  operational_duty_cycle?: number | null;
  /**
   * Optional per-job overrides for the schema-v9 payload mission
   * requirements. The NSGA-II candidates are all scored carrying this
   * payload, so the front reflects the mission's real mass/power budget.
   */
  payload_mass_kg?: number | null;
  payload_power_w?: number | null;
}

export interface OptimizeJobResponse {
  job_id: string;
  status: OptimizeJobStatus;
  stream_url: string;
  result_url: string;
  cancel_url: string;
}

export interface OptimizeCheckpointOut {
  gen: number;
  hypervolume: number;
  pareto_size: number;
  best_per_objective: Record<string, number>;
}

export interface OptimizeParetoPoint {
  design: DesignVector;
  metrics: Record<PrimaryTarget, number>;
}

export interface OptimizeResultResponse {
  job_id: string;
  status: OptimizeJobStatus;
  backend_used: OptimizeBackend | null;
  checkpoints: OptimizeCheckpointOut[];
  pareto_front: OptimizeParetoPoint[];
  error: string | null;
}

export interface OptimizeCancelResponse {
  job_id: string;
  status: OptimizeJobStatus;
}

export interface ShapFeatureScore {
  feature: string;
  value: number;
}

export interface ShapExplainRequest {
  design: DesignVector;
  scenario_name: string;
  target: PrimaryTarget;
  operational_duty_cycle?: number | null;
  payload_mass_kg?: number | null;
  payload_power_w?: number | null;
}

export interface ShapLocalResponse {
  target: PrimaryTarget;
  prediction: number;
  base_value: number;
  contributions: ShapFeatureScore[];
}

export interface RegistryEntrySummary {
  rover_name: string;
  is_flown: boolean;
  design: DesignVector;
  scenario: MissionScenario;
  gravity_m_per_s2: number;
  thermal_architecture: Record<string, unknown>;
  panel_efficiency: number;
  panel_dust_factor: number;
  panel_tilt_deg: number;
  panel_azimuth_deg: number;
  imputation_notes: string;
}

export interface RegistryListResponse {
  rovers: RegistryEntrySummary[];
}

/**
 * Mirror of the FastAPI `RediscoveryBackend` literal. The evaluator
 * sweep is the source of truth (corrected-evaluator-truth fronts);
 * the v8 surrogate sweep is offered as a wall-clock benchmark
 * alongside it. See `reports/rediscovery_loo_comparison.md` for the
 * 2026-05-28 panel-tilt fix that makes the polar trio's energy
 * margins physically defensible under the evaluator backend.
 */
export type RediscoveryBackend = "evaluator" | "surrogate";

export interface RediscoveryParetoPoint {
  design: DesignVector;
  metrics: Record<string, number>;
}

export interface RediscoverySummary {
  /** URL-safe identifier matching the per-rover JSON file stem. */
  slug: string;
  /** Human-readable name from the registry (e.g. `"CADRE-unit"`). */
  rover_name: string;
  /** True for rovers that have actually flown (Pragyan, Yutu-2). */
  is_flown: boolean;
  /** Class-generic `*_micro` scenario the rediscovery harness used. */
  class_generic_scenario: string;
  backend: RediscoveryBackend;
  design_space_distance: number;
  pareto_dominated: boolean;
  mass_budget_kg: number;
  pareto_front_size: number;
}

export interface RediscoveryListResponse {
  backend: RediscoveryBackend;
  rovers: RediscoverySummary[];
}

export interface RediscoveryDetail {
  slug: string;
  rover_name: string;
  is_flown: boolean;
  class_generic_scenario: string;
  backend: RediscoveryBackend;
  rover_design: DesignVector;
  rover_metrics_under_generic_scenario: Record<string, number>;
  design_space_distance: number;
  pareto_dominated: boolean;
  mass_budget_kg: number;
  integer_matches: Record<string, boolean>;
  per_variable_percent_errors: Record<string, number>;
  nearest_pareto_index: number;
  nearest_pareto_design: DesignVector;
  nearest_pareto_metrics: Record<string, number>;
  pareto_front: RediscoveryParetoPoint[];
}

/**
 * Static design-space bounds, kept aligned with
 * `roverdevkit/schema.py::DesignVector`. The form uses these for
 * range validation, slider extents, and step sizes; if the Python
 * schema bounds change we update them here too (caught at runtime
 * by FastAPI's 422 response, but a same-day visual diff is nicer).
 */
export interface FieldBounds {
  min: number;
  max: number;
  step: number;
  unit: string;
  label: string;
  description: string;
}

export const DESIGN_BOUNDS: Record<keyof DesignVector, FieldBounds> = {
  wheel_radius_m: {
    min: 0.05,
    max: 0.2,
    step: 0.005,
    unit: "m",
    label: "Wheel radius",
    description: "R, mobility wheel radius.",
  },
  wheel_width_m: {
    min: 0.03,
    max: 0.2,
    step: 0.005,
    unit: "m",
    label: "Wheel width",
    description: "W, mobility wheel width.",
  },
  grouser_height_m: {
    min: 0.0,
    max: 0.02,
    step: 0.001,
    unit: "m",
    label: "Grouser height",
    description: "h_g, soil-engaging tooth height.",
  },
  grouser_count: {
    min: 0,
    max: 24,
    step: 1,
    unit: "",
    label: "Grouser count",
    description: "N_g, grousers per wheel.",
  },
  n_wheels: {
    min: 4,
    max: 6,
    step: 2,
    unit: "",
    label: "Wheel count",
    description: "N_w, mobility wheel count (4 or 6).",
  },
  chassis_mass_kg: {
    min: 0.5,
    max: 50,
    step: 0.1,
    unit: "kg",
    label: "Chassis mass",
    description:
      "m_c, dry chassis mass (subsystem masses are added by the model). Floor lowered to 0.5 kg to admit CADRE / Tenacious-class ultra-micro rovers; designs below 3 kg are outside the v4 surrogate's training support and fall back to the deterministic evaluator (no PIs).",
  },
  wheelbase_m: {
    min: 0.3,
    max: 1.2,
    step: 0.05,
    unit: "m",
    label: "Wheelbase",
    description: "L_wb, longitudinal wheel separation.",
  },
  solar_area_m2: {
    min: 0.1,
    max: 1.5,
    step: 0.05,
    unit: "m^2",
    label: "Solar area",
    description: "A_s, deployable solar array area.",
  },
  battery_capacity_wh: {
    min: 5,
    max: 500,
    step: 1,
    unit: "Wh",
    label: "Battery capacity",
    description:
      "C_b, usable battery capacity. Floor lowered to 5 Wh to admit CADRE-class flotilla rovers; designs below 20 Wh are outside the v4 surrogate's training support.",
  },
  avionics_power_w: {
    min: 5,
    max: 40,
    step: 0.5,
    unit: "W",
    label: "Avionics power",
    description: "P_a, continuous avionics draw.",
  },
  peak_wheel_torque_nm: {
    min: 0.05,
    max: 20.0,
    step: 0.01,
    unit: "Nm",
    label: "Peak wheel torque",
    description:
      "T_hub^peak, peak per-wheel hub torque the drivetrain (motor + gearbox combined) can sustain. Sizes motor mass and gates whether the rover stalls on slope; cruise speed is derived from this and the slip-balance torque demand. Floor lowered to 0.05 Nm to admit CADRE-class flotilla rovers; designs below 0.3 Nm are outside the v4 surrogate's training support.",
  },
};

/**
 * Bounds for the schema-v9 payload mission-requirement sliders. Kept
 * aligned with `MissionScenario.payload_mass_kg` / `payload_power_w`
 * (both `[0, 30]`). The 30 kg ceiling covers the heaviest in-class lunar
 * micro-rover payload (Yutu-2, ~25 kg of GPR/VNIS/APXS instruments).
 */
export const PAYLOAD_BOUNDS: Record<
  "payload_mass_kg" | "payload_power_w",
  FieldBounds
> = {
  payload_mass_kg: {
    min: 0,
    max: 30,
    step: 0.5,
    unit: "kg",
    label: "Payload mass",
    description:
      "m_payload, scientific-instrument mass. A mission requirement added to total vehicle mass outside the dry-mass growth margin.",
  },
  payload_power_w: {
    min: 0,
    max: 30,
    step: 0.5,
    unit: "W",
    label: "Payload power",
    description:
      "P_payload, continuous ops-time instrument power. Adds to the electrical load alongside avionics in the traverse budget.",
  },
};

/** User-facing display metadata for the four predicted performance metrics. */
export const TARGET_META: Record<
  PrimaryTarget,
  { label: string; unit: string; description: string }
> = {
  range_km: {
    label: "Range",
    unit: "km",
    description:
      "Distance the rover can traverse during the scenario at its commanded drive duty cycle.",
  },
  energy_margin_raw_pct: {
    label: "Energy margin",
    unit: "%",
    description:
      "(Energy generated − energy used) / energy used over the traverse. Positive values mean surplus solar generation; large positive values are typical for energy-rich designs.",
  },
  slope_capability_deg: {
    label: "Slope capability",
    unit: "deg",
    description:
      "Steepest slope the rover can sustain on the scenario's soil at its commanded speed.",
  },
  total_mass_kg: {
    label: "Total mass",
    unit: "kg",
    description:
      "Estimated dry mass of the rover, including chassis, motors, structure, power, and avionics.",
  },
};
