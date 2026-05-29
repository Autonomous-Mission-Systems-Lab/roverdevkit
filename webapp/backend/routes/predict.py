"""``POST /predict`` — surrogate point prediction with 90 % PI."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import (
    FeatureRow,
    PredictRequest,
    PredictResponse,
    PredictTarget,
)
from webapp.backend.services import apply_scenario_overrides
from webapp.backend.services.predict import build_feature_row, predict_quantiles

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return median + 90 % prediction intervals for the four primary targets.

    Pipeline
    --------
    1. Resolve the scenario from the canonical four (404 if unknown).
    2. Look up nominal Bekker-Wong soil parameters for the scenario's
       simulant.
    3. Assemble the 27-D feature row in the surrogate's training-time
       column order, applying any per-call ``operational_duty_cycle``
       and schema-v9 payload (``payload_mass_kg`` / ``payload_power_w``)
       mission-requirement overrides before flattening so the surrogate
       sees the same scenario inputs the deterministic evaluator would.
    4. Dispatch to every primary target's ``QuantileHeads`` head and
       collect ``(q05, q50, q95)`` triples.

    The surrogate is the quantile-calibration ``quantile_bundles.joblib``;
    ``q50`` is within R² 0.005 of the tuned-median tuned median (see
    ``reports/intervals_v4/SUMMARY.md`` for the median sanity
    guardrail), so this single artifact powers both point estimates
    and PI envelopes.

    Schema v7_1 (v7_1 schema follow-on): ``operational_duty_cycle`` is
    a true surrogate input feature (LHS-sampled per row over [0, 0.6]),
    so any in-bounds δ_ops is in-distribution and the calibrated PIs
    apply across the full frontend slider range. The pre-v7_1
    "evaluator-only fallback when override differs from default" gate
    has been removed; ``mode`` is always ``"surrogate"``.
    """
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
    )
    soil = get_soil_for_simulant(scenario.soil_simulant)

    X = build_feature_row(req.design, scenario, soil)

    try:
        bundles = get_quantile_bundles()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "surrogate artifact not loaded; run scripts/calibrate_intervals.py first."
            ),
        ) from exc
    preds = predict_quantiles(bundles, X, repair_crossings=req.repair_crossings)
    targets = [
        PredictTarget(
            target=t,  # type: ignore[arg-type]
            q05=preds[t]["q05"],
            q50=preds[t]["q50"],
            q95=preds[t]["q95"],
        )
        for t in PRIMARY_REGRESSION_TARGETS
    ]

    feature_row = FeatureRow(
        columns=list(X.columns),
        values=[v.item() if hasattr(v, "item") else v for v in X.iloc[0].tolist()],
    )

    return PredictResponse(
        scenario_name=req.scenario_name,
        predictions=targets,
        feature_row=feature_row,
        mode="surrogate",
    )
