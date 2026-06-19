"""``POST /evaluate`` — deterministic analytical mission evaluator.

This is the *single-shot* counterpart to ``/predict``. It runs the same
physics pipeline that produced the surrogate's training corpus, so the
returned values are the ground truth the surrogate is regressing
against. The single-design panel uses ``/evaluate`` for the median
value of each metric (and for real-rover overlays) and ``/predict``
only for the surrogate's calibrated 90 % prediction-interval band.

The analytical evaluator runs in ~30 ms after the traverse-loop
lift-out, which is imperceptible for one-click UX. The 50k+-evaluation
inner loops (NSGA-II, feasibility heatmaps) keep using the surrogate
because even 30 ms × 50k is ~25 minutes of wall-clock.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from roverdevkit.surrogate.features import PRIMARY_REGRESSION_TARGETS
from webapp.backend.loaders import get_canonical_scenarios
from webapp.backend.services import apply_scenario_overrides
from webapp.backend.schemas import (
    ArchitectureDiagnosticOut,
    EvaluateMetric,
    EvaluateRequest,
    EvaluateResponse,
    StallDiagnosticOut,
    ThermalDiagnosticOut,
)
from webapp.backend.services.evaluate import (
    evaluate_design,
    metrics_as_primary_dict,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_route(req: EvaluateRequest) -> EvaluateResponse:
    """Run the analytical mission evaluator on one design × one scenario.

    Pipeline
    --------
    1. Resolve the scenario from the canonical four (404 if unknown).
    2. Dispatch to :func:`roverdevkit.mission.evaluator.evaluate_verbose`.
    3. Project ``MissionMetrics`` onto the four primary targets and
       attach structured ``thermal`` / ``stall`` diagnostics (schema
       v6) plus the runtime-resolved ``effective_duty_cycle`` and
       ``cruise_speed_mps`` so the panel chip can explain *why* a
       survival flag fired.
    """
    scenarios = get_canonical_scenarios()
    if req.scenario_name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown scenario {req.scenario_name!r}. "
                f"Pick one of {sorted(scenarios.keys())}."
            ),
        )
    scenario = apply_scenario_overrides(
        scenarios[req.scenario_name],
        payload_mass_kg=req.payload_mass_kg,
        payload_power_w=req.payload_power_w,
        mission_duration_earth_days=req.mission_duration_earth_days,
        required_obstacle_height_m=req.required_obstacle_height_m,
    )

    output = evaluate_design(
        req.design,
        scenario,
        operational_duty_cycle=req.operational_duty_cycle,
        required_obstacle_height_m=req.required_obstacle_height_m,
    )
    primary = metrics_as_primary_dict(output.metrics)

    metrics = [
        EvaluateMetric(target=t, value=primary[t])  # type: ignore[arg-type]
        for t in PRIMARY_REGRESSION_TARGETS
    ]

    arch = output.thermal  # ThermalResult
    thermal_out = ThermalDiagnosticOut(
        survives=bool(arch.survives),
        peak_sun_temp_c=float(arch.peak_sun_temp_c),
        lunar_night_temp_c=float(arch.lunar_night_temp_c),
        # The default architecture used by the evaluator pins these
        # limits at -30 / +50 °C; we re-state them here so the frontend
        # never has to hardcode a number.
        min_operating_temp_c=-30.0,
        max_operating_temp_c=50.0,
        rhu_power_w=0.0,
        hibernation_power_w=2.0,
        # Surface area is rebuilt from the chassis mass via the same
        # cube-root proxy used inside `evaluate_verbose`; we reproduce
        # it for the response so the dialog can show users what
        # radiating area the model assumed.
        surface_area_m2=0.02 * (req.design.chassis_mass_kg ** (2.0 / 3.0)) + 0.05,
        hot_case_ok=arch.peak_sun_temp_c <= 50.0,
        cold_case_ok=arch.lunar_night_temp_c >= -30.0,
    )

    st = output.stall
    stall_out = StallDiagnosticOut(
        stalled=bool(st.stalled),
        peak_torque_demand_nm=float(st.peak_torque_demand_nm),
        peak_torque_capacity_nm=float(st.peak_torque_capacity_nm),
    )

    return EvaluateResponse(
        scenario_name=req.scenario_name,
        metrics=metrics,
        thermal=thermal_out,
        stall=stall_out,
        architecture=ArchitectureDiagnosticOut(
            mobility_architecture=req.design.mobility_architecture,
            obstacle_capability_m=float(output.metrics.obstacle_capability_m),
            required_obstacle_height_m=float(
                req.required_obstacle_height_m
                if req.required_obstacle_height_m is not None
                else scenario.required_obstacle_height_m
            ),
            obstacle_margin_m=float(output.metrics.obstacle_margin_m),
            obstacle_requirement_met=bool(output.metrics.obstacle_requirement_met),
            architecture_mass_kg=float(output.metrics.architecture_mass_kg),
        ),
        effective_duty_cycle=float(output.effective_duty_cycle),
        cruise_speed_mps=float(output.cruise_speed_mps),
        elapsed_ms=output.elapsed_ms,
    )
