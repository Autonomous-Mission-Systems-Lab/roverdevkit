"""SHAP/design-rule explanation routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from roverdevkit.surrogate.uncertainty import QuantileHeads
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import (
    ShapExplainRequest,
    ShapFeatureScore,
    ShapGlobalResponse,
    ShapLocalResponse,
    ShapTargetImportance,
)
from webapp.backend.services.predict import build_feature_row

router = APIRouter(tags=["shap"])


@router.get("/shap/global", response_model=ShapGlobalResponse)
def shap_global() -> ShapGlobalResponse:
    """Return global feature importances from the median quantile heads."""
    bundles = _load_bundles_or_503()
    rows: list[ShapTargetImportance] = []
    for target in PRIMARY_REGRESSION_TARGETS:
        head = bundles[target]
        model = _median_model(head)
        values = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
        if values.size != len(head.feature_columns):
            values = np.zeros(len(head.feature_columns), dtype=float)
        denom = float(np.sum(np.abs(values))) or 1.0
        scores = [
            ShapFeatureScore(feature=feature, value=float(value / denom))
            for feature, value in zip(head.feature_columns, values, strict=True)
        ]
        rows.append(
            ShapTargetImportance(
                target=target,  # type: ignore[arg-type]
                features=sorted(scores, key=lambda item: abs(item.value), reverse=True)[:12],
            )
        )
    return ShapGlobalResponse(targets=rows)


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
    scenario = scenarios[req.scenario_name]
    if req.operational_duty_cycle is not None:
        scenario = scenario.model_copy(
            update={"operational_duty_cycle": req.operational_duty_cycle}
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
        # exact TreeSHAP contributions. The global panel still carries
        # the explanatory signal and the response shape remains stable.
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
