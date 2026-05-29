"""Feature-row construction and surrogate dispatch.

The single public entry point is :func:`predict_for_design`, which
mirrors the row-flattening logic in :mod:`roverdevkit.surrogate.dataset`
so the live API and the training pipeline produce *bit-identical*
input rows. Sharing the column order from
:data:`roverdevkit.surrogate.features.INPUT_COLUMNS` is what makes that
guarantee tractable.

Why not import the dataset flatteners directly
----------------------------------------------
The training-time flatteners take an :class:`LHSSample` (which carries
the *jittered* soil parameters, scenario_family, etc.). At inference
time we have only a (design, scenario) pair; the soil parameters come
from the catalogue's nominal values, and ``scenario_family`` is
synthesised from the canonical scenario name (which is exactly how the
LHS sampler picks it -- see ``surrogate/sampling.py`` line 392). Doing
the construction here keeps that mapping in one place rather than
forcing the dataset module to grow an "inference mode".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.features import (
    INPUT_COLUMNS,
    PRIMARY_REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
)
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.bekker_wong import SoilParameters


def build_feature_row(
    design: DesignVector,
    scenario: MissionScenario,
    soil: SoilParameters,
    *,
    scenario_family: str | None = None,
) -> pd.DataFrame:
    """Build the 27-column input frame the surrogate expects.

    SCHEMA_VERSION v7_1: ``scenario_operational_duty_cycle`` is now a
    surrogate input column (it became a per-row LHS feature in v7_1).
    The caller must therefore pass the *effective* δ_ops -- usually
    ``scenario.operational_duty_cycle`` after applying any per-call
    override -- on the scenario object so the surrogate sees the
    same δ_ops the deterministic evaluator would use.

    Parameters
    ----------
    design
        Validated design vector. The ``DesignVector`` schema's own
        bounds are the only place input ranges are enforced; callers
        should rely on Pydantic to reject out-of-bounds requests.
    scenario
        The canonical mission scenario (loaded from YAML).
    soil
        Nominal Bekker-Wong soil parameters for ``scenario.soil_simulant``.
    scenario_family
        Categorical family label. Defaults to ``scenario.name``, which
        matches how the LHS sampler tags rows for the canonical four
        scenarios.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with columns in :data:`INPUT_COLUMNS`
        order; the four categorical columns have ``category`` dtype.
    """
    family = scenario_family if scenario_family is not None else scenario.name
    row: dict[str, Any] = {
        # Design (11) — schema v7 dropped designed_duty_cycle
        "design_wheel_radius_m": design.wheel_radius_m,
        "design_wheel_width_m": design.wheel_width_m,
        "design_grouser_height_m": design.grouser_height_m,
        "design_grouser_count": int(design.grouser_count),
        "design_n_wheels": int(design.n_wheels),
        "design_chassis_mass_kg": design.chassis_mass_kg,
        "design_wheelbase_m": design.wheelbase_m,
        "design_solar_area_m2": design.solar_area_m2,
        "design_battery_capacity_wh": design.battery_capacity_wh,
        "design_avionics_power_w": design.avionics_power_w,
        "design_peak_wheel_torque_nm": design.peak_wheel_torque_nm,
        # Scenario numerics (10) — v7_1 promoted operational_duty_cycle
        # to a true surrogate input feature.
        "scenario_latitude_deg": scenario.latitude_deg,
        "scenario_mission_duration_earth_days": scenario.mission_duration_earth_days,
        "scenario_max_slope_deg": scenario.max_slope_deg,
        "scenario_operational_duty_cycle": scenario.operational_duty_cycle,
        "scenario_soil_n": soil.n,
        "scenario_soil_k_c": soil.k_c,
        "scenario_soil_k_phi": soil.k_phi,
        "scenario_soil_cohesion_kpa": soil.cohesion_kpa,
        "scenario_soil_friction_angle_deg": soil.friction_angle_deg,
        "scenario_soil_shear_modulus_k_m": soil.shear_modulus_k_m,
        # Payload mission requirements (schema v9) — sampled
        # family-agnostic uniform [0, 30] in the LHS, so the webapp
        # Mission-Inputs sliders stay in-distribution.
        "scenario_payload_mass_kg": scenario.payload_mass_kg,
        "scenario_payload_power_w": scenario.payload_power_w,
        # Scenario categoricals (4)
        "scenario_family": family,
        "scenario_terrain_class": scenario.terrain_class,
        "scenario_soil_simulant": scenario.soil_simulant,
        "scenario_sun_geometry": scenario.sun_geometry,
    }
    df = pd.DataFrame([row], columns=INPUT_COLUMNS)
    for col in SCENARIO_CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def predict_quantiles(
    bundles: dict[str, QuantileHeads],
    X: pd.DataFrame,
    *,
    repair_crossings: bool = True,
) -> dict[str, dict[str, float]]:
    """Run every primary-target quantile head on ``X`` and return a flat dict.

    Parameters
    ----------
    bundles
        Output of :func:`webapp.backend.loaders.get_quantile_bundles`.
    X
        Single-row feature frame from :func:`build_feature_row`.
    repair_crossings
        Sort the (q05, q50, q95) triple per row so the response is
        always monotone. See ``surrogate/uncertainty.py`` for why this
        is safe.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{target: {"q05": ..., "q50": ..., "q95": ...}}``. Iteration
        order matches :data:`PRIMARY_REGRESSION_TARGETS` so the
        frontend can render rows deterministically.

    Raises
    ------
    KeyError
        If any primary target is missing from ``bundles``. We surface
        the full diff so a stale joblib file is easy to diagnose.
    """
    missing = [t for t in PRIMARY_REGRESSION_TARGETS if t not in bundles]
    if missing:
        raise KeyError(
            f"quantile bundles missing primary targets: {missing}. "
            "Re-run scripts/calibrate_intervals.py."
        )
    out: dict[str, dict[str, float]] = {}
    for target in PRIMARY_REGRESSION_TARGETS:
        head = bundles[target]
        preds = head.predict(X, repair_crossings=repair_crossings)
        out[target] = {k: float(v[0]) for k, v in preds.items()}
    return out
