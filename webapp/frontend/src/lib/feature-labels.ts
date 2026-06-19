/**
 * Human-readable labels for surrogate input features (SHAP, feature rows).
 *
 * Keys match `roverdevkit.surrogate.features.INPUT_COLUMNS` — the 27-D
 * design × scenario vector fed to the quantile heads.
 */
export const SURROGATE_FEATURE_LABELS: Record<string, string> = {
  // Design vector
  design_wheel_radius_m: "Wheel radius",
  design_wheel_width_m: "Wheel width",
  design_grouser_height_m: "Grouser height",
  design_grouser_count: "Grouser count",
  design_n_wheels: "Wheel count",
  design_chassis_mass_kg: "Chassis mass",
  design_wheelbase_m: "Wheelbase",
  design_solar_area_m2: "Solar array area",
  design_battery_capacity_wh: "Battery capacity",
  design_avionics_power_w: "Avionics power",
  design_peak_wheel_torque_nm: "Peak wheel torque",

  // Scenario — continuous
  scenario_latitude_deg: "Landing latitude",
  scenario_mission_duration_earth_days: "Mission duration",
  scenario_max_slope_deg: "Scenario max slope",
  scenario_operational_duty_cycle: "Operational duty cycle",
  scenario_soil_n: "Soil bearing exponent (n)",
  scenario_soil_k_c: "Soil cohesion modulus (k_c)",
  scenario_soil_k_phi: "Soil friction modulus (k_φ)",
  scenario_soil_cohesion_kpa: "Soil cohesion",
  scenario_soil_friction_angle_deg: "Soil friction angle",
  scenario_soil_shear_modulus_k_m: "Soil shear modulus (K)",
  scenario_payload_mass_kg: "Payload mass",
  scenario_payload_power_w: "Payload power",

  // Scenario — categorical
  scenario_family: "Mission scenario type",
  scenario_terrain_class: "Terrain class",
  scenario_soil_simulant: "Soil simulant",
  scenario_sun_geometry: "Sun geometry",
};

/** Map a raw surrogate feature column name to a UI label. */
export function formatFeatureLabel(feature: string): string {
  const mapped = SURROGATE_FEATURE_LABELS[feature];
  if (mapped) return mapped;

  // Fallback for unexpected / legacy column names.
  return feature
    .replace(/^design_/, "")
    .replace(/^scenario_/, "Scenario ")
    .replace(/_/g, " ")
    .replace(/\bdeg\b/g, "angle")
    .replace(/\bm2\b/g, "area")
    .replace(/\bwh\b/g, "capacity")
    .replace(/\bkg\b/g, "mass")
    .replace(/\bw\b/g, "power")
    .replace(/\bnm\b/g, "torque")
    .replace(/\bkpa\b/g, "kPa")
    .trim();
}
