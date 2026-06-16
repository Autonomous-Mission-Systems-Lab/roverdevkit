"""Per-design SHAP explanation route.

Returns TreeSHAP-style feature contributions for the current design and the
selected primary target. There is intentionally no global-importance
endpoint: the design-explain experience in the webapp is scoped to the
single design the user is editing, and any global feature-importance
analysis lives in the offline design-rules report under ``reports/``.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from roverdevkit.surrogate.uncertainty import QuantileHeads
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import (
    ShapExplainRequest,
    ShapFeatureScore,
    ShapLocalResponse,
)
from webapp.backend.services import apply_scenario_overrides
from webapp.backend.services.predict import build_feature_row

router = APIRouter(tags=["shap"])


@router.post("/shap/explain", response_model=ShapLocalResponse)
def shap_explain(req: ShapExplainRequest) -> ShapLocalResponse:
    """Return per-feature contributions for the current design and target."""
    bundles = _load_bundles_or_503()
    scenarios = get_canonical_scenarios()
    if req.scenario_name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown scenario {req.scenario_name!r}. Pick one of {sorted(scenarios.keys())}."
            ),
        )
    scenario = apply_scenario_overrides(
        scenarios[req.scenario_name],
        operational_duty_cycle=req.operational_duty_cycle,
        payload_mass_kg=req.payload_mass_kg,
        payload_power_w=req.payload_power_w,
        mission_duration_earth_days=req.mission_duration_earth_days,
    )
    soil = get_soil_for_simulant(scenario.soil_simulant)
    X = build_feature_row(req.design, scenario, soil)
    head = bundles[req.target]
    model = _median_model(head)
    X_aligned = X[list(head.feature_columns)]
    prediction = float(np.asarray(model.predict(X_aligned))[0])

    base_value = 0.0
    contrib_values = np.zeros(len(head.feature_columns), dtype=float)
    try:
        import xgboost as xgb

        dmat = xgb.DMatrix(X_aligned, enable_categorical=True)
        contribs = np.asarray(model.get_booster().predict(dmat, pred_contribs=True))[0]
        contrib_values = contribs[:-1]
        base_value = float(contribs[-1])
    except Exception:
        # Keep the UI usable even if a future model backend cannot emit
        # exact TreeSHAP contributions. The response shape stays stable
        # and the chart simply falls back to a flat zero-contribution row.
        base_value = prediction

    scores = [
        ShapFeatureScore(feature=feature, value=float(value))
        for feature, value in zip(head.feature_columns, contrib_values, strict=True)
    ]
    return ShapLocalResponse(
        target=req.target,
        prediction=prediction,
        base_value=base_value,
        contributions=sorted(scores, key=lambda item: abs(item.value), reverse=True)[:12],
    )


def _load_bundles_or_503() -> dict[str, QuantileHeads]:
    try:
        return get_quantile_bundles()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="surrogate artifact not loaded; run scripts/calibrate_intervals.py first.",
        ) from exc


def _median_model(head: QuantileHeads):
    idx = min(range(len(head.quantiles)), key=lambda i: abs(head.quantiles[i] - 0.5))
    return head.models[idx]
