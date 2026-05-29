"""Feature-matrix construction and column inventories for the surrogate.

This module is the single source of truth for **which columns the
baseline and multi-fidelity surrogates train on**.
It mirrors the flat Parquet schema emitted by
:mod:`roverdevkit.surrogate.dataset` (see ``data/analytical/SCHEMA.md``)
and intentionally takes no ML-library dependency so the mission
evaluator can import it transitively without pulling XGBoost / sklearn.

Columns
-------
Inputs (27 columns):

- :data:`DESIGN_FEATURE_COLUMNS` (11) — the raw design vector.
- :data:`SCENARIO_NUMERIC_COLUMNS` (12) — continuous scenario + soil
  parameters (latitude, mission duration, max slope, ground-ops duty
  cycle, Bekker n / k_c / k_phi / cohesion / friction / shear modulus,
  payload mass, payload power).
- :data:`SCENARIO_CATEGORICAL_COLUMNS` (4) — scenario-family discrete
  features. Kept as pandas ``category`` dtype so XGBoost can consume
  them natively via ``enable_categorical=True`` without one-hot
  blow-up.

Targets:

- :data:`REGRESSION_TARGETS` — the 7 numeric mission metrics. The
  primary ones (range, raw energy margin, slope, total mass) are what
  the baseline-surrogate accuracy table reports on; the others are secondary
  diagnostics.
- :data:`CLASSIFICATION_TARGETS` — the single ``motor_torque_ok``
  feasibility flag (a real Bekker-Wong outcome that depends on grouser
  geometry, soil shear parameters, slope, and mass).

Why no thermal target
---------------------
``thermal_survival`` was dropped from the surrogate schema in v2 (see
``data/analytical/SCHEMA.md``): with the current mass model RHU power
and MLI quality are free, so thermal reduces to a near-trivial gate
without a real design trade-off. The system-level evaluator still
computes it as a diagnostic; a future mass-model upgrade that charges
RHU/MLI mass would restore thermal as a learnable Pareto target.

Engineered features (``add_engineered_features``) are deferred to
SCM-correction: base numeric + categorical columns are sufficient for the
baseline-surrogate XGBoost baseline and the multi-fidelity composition, and
adding engineered features pre-baseline would confound the
"did-features-help?" ablation.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Input columns
# ---------------------------------------------------------------------------

DESIGN_FEATURE_COLUMNS: list[str] = [
    "design_wheel_radius_m",
    "design_wheel_width_m",
    "design_grouser_height_m",
    "design_grouser_count",
    "design_n_wheels",
    "design_chassis_mass_kg",
    "design_wheelbase_m",
    "design_solar_area_m2",
    "design_battery_capacity_wh",
    "design_avionics_power_w",
    "design_peak_wheel_torque_nm",
]
"""11-D design vector, prefixed to match the Parquet schema.

SCHEMA_VERSION v6 (v6 schema update): ``design_nominal_speed_mps`` ->
``design_peak_wheel_torque_nm`` (cruise speed is now derived inside
the evaluator, no longer a free design input);
``design_drive_duty_cycle`` -> ``design_designed_duty_cycle``.

SCHEMA_VERSION v7 (v7 schema follow-up): drops
``design_designed_duty_cycle`` after that field turned out to do no
engineering work in the v6 mass model. Drive duty cycle lives on the
scenario (``scenario_operational_duty_cycle``) only; per-call
overrides are wired through :class:`MissionScenario` at inference."""

SCENARIO_NUMERIC_COLUMNS: list[str] = [
    "scenario_latitude_deg",
    "scenario_mission_duration_earth_days",
    "scenario_max_slope_deg",
    "scenario_operational_duty_cycle",
    "scenario_soil_n",
    "scenario_soil_k_c",
    "scenario_soil_k_phi",
    "scenario_soil_cohesion_kpa",
    "scenario_soil_friction_angle_deg",
    "scenario_soil_shear_modulus_k_m",
    "scenario_payload_mass_kg",
    "scenario_payload_power_w",
]
"""Continuous scenario + jittered Bekker-soil inputs (12 columns).

``scenario_traverse_distance_m`` is intentionally excluded: it is
family-fixed (non-binding) and would otherwise leak the scenario
identity into a supposedly continuous feature.

SCHEMA_VERSION v7_1 (v7_1 schema follow-on, 2026-04-28): added
``scenario_operational_duty_cycle`` so the surrogate sees δ_ops as a
true continuous input. Pre-v7_1 the surrogate keyed off the
family categorical only, which made calibrated PIs available only
at the four per-family δ_ops anchors and forced the webapp to fall
through to evaluator-only mode whenever the user moved the δ_ops
slider away from its default. With δ_ops now an LHS feature
(uniform [0, 0.6] independently of family), the calibrated quantile
heads cover the full slider range.

SCHEMA_VERSION v9 (payload as a mission requirement): added
``scenario_payload_mass_kg`` and ``scenario_payload_power_w`` so the
surrogate sees scientific payload as true continuous inputs. Both are
sampled family-agnostic uniform on [0, 30] (see
:data:`roverdevkit.surrogate.sampling._PAYLOAD_MASS_KG_BOUNDS`), so the
webapp Mission-Inputs payload sliders stay in-distribution for
calibrated PIs — same rationale as the v7_1 δ_ops promotion."""

SCENARIO_CATEGORICAL_COLUMNS: list[str] = [
    "scenario_family",
    "scenario_terrain_class",
    "scenario_soil_simulant",
    "scenario_sun_geometry",
]
"""Scenario-family categorical inputs (4 columns). Keep as pandas
``category`` dtype and let XGBoost handle them natively via
``enable_categorical=True`` rather than one-hot encoding."""

INPUT_COLUMNS: list[str] = (
    DESIGN_FEATURE_COLUMNS + SCENARIO_NUMERIC_COLUMNS + SCENARIO_CATEGORICAL_COLUMNS
)
"""Concatenated input column list used by :func:`build_feature_matrix`."""


# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------

REGRESSION_TARGETS: list[str] = [
    "range_km",
    "energy_margin_pct",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "peak_motor_torque_nm",
    "sinkage_max_m",
]
"""All numeric mission-metric targets."""

PRIMARY_REGRESSION_TARGETS: list[str] = [
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
]
"""Primary target subset used for surrogate accuracy reporting."""

CLASSIFICATION_TARGETS: list[str] = ["stalled"]
"""Single binary feasibility target.

SCHEMA_VERSION v6 (v6 schema update): switched from ``motor_torque_ok`` to
``stalled`` to match the new explicit drivetrain stall gate (per-wheel
torque demand vs ``DesignVector.peak_wheel_torque_nm``). ``stalled``
is the *infeasible* class (1 = stalled), inverted from
``motor_torque_ok``'s OK-is-1 convention; baseline / classifier
training scripts re-bind class-weight semantics accordingly. The
underlying physics-driven binary outcome is unchanged: it captures
whether the rover can generate enough drawbar pull (and has enough
torque envelope) to climb the scenario's worst-case slope under
Bekker-Wong terramechanics with the sampled soil parameters."""

FEASIBILITY_COLUMN: str = "stalled"
"""Alias for the single feasibility column. Kept as a constant so
downstream baselines / classifiers reference one canonical name even
if the underlying definition changes (e.g. when thermal is restored as
a real trade-off in a future mass-model upgrade). Schema v6 flipped
this from ``motor_torque_ok`` to ``stalled``; positive class is now
the failure mode (= ``run_traverse(...).rover_stalled``)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-input columns in canonical order.

    The returned DataFrame is a shallow copy of ``df`` restricted to
    :data:`INPUT_COLUMNS`; categorical dtypes are preserved for direct
    XGBoost ``enable_categorical=True`` consumption. Callers should
    filter to ``status == 'ok'`` rows (see :func:`valid_rows`) before
    passing to the trainer -- rows where the evaluator raised have NaN
    targets and this helper does **not** drop them.

    Raises
    ------
    KeyError
        If any required column is missing, with the full missing list
        in the message so a stale Parquet file is easy to diagnose.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing required input columns: {missing}. "
            "Check SCHEMA_VERSION on the source Parquet."
        )
    return df.loc[:, INPUT_COLUMNS].copy()


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows where the evaluator succeeded.

    Drops any row with ``status != 'ok'`` (or missing ``status``). Also
    drops rows where primary regression targets are NaN, which can
    happen if a physics sub-model silently returns non-finite values.
    """
    if "status" in df.columns:
        mask = df["status"].astype(str) == "ok"
    else:
        mask = pd.Series(True, index=df.index)
    for col in PRIMARY_REGRESSION_TARGETS:
        if col in df.columns:
            mask &= df[col].notna()
    return df.loc[mask].copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns appended.

    Placeholder: the baseline uses only raw + categorical features; engineered
    features should be introduced in a dedicated ablation.
    """
    raise NotImplementedError(
        "Engineered feature generation is not implemented yet."
    )


__all__ = [
    "CLASSIFICATION_TARGETS",
    "DESIGN_FEATURE_COLUMNS",
    "FEASIBILITY_COLUMN",
    "INPUT_COLUMNS",
    "PRIMARY_REGRESSION_TARGETS",
    "REGRESSION_TARGETS",
    "SCENARIO_CATEGORICAL_COLUMNS",
    "SCENARIO_NUMERIC_COLUMNS",
    "add_engineered_features",
    "build_feature_matrix",
    "valid_rows",
]
