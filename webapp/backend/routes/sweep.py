"""``POST /sweep`` — 1-D or 2-D parametric sweep.

The frontend sends a base design, a scenario, one (or two) sweep
axes, a target metric, and a backend choice. We return the chosen
target's value over the whole grid plus enough metadata for the
client to render a Plotly line / heatmap without further math.

Backend dispatch
----------------
- ``backend="auto"``: corrected evaluator below
  :data:`~roverdevkit.tradespace.sweeps.EVALUATOR_AUTO_THRESHOLD` cells,
  surrogate otherwise. The default for the UI.
- ``backend="evaluator"`` / ``backend="surrogate"``: forced; capped
  by the per-backend hard limits in
  :mod:`roverdevkit.tradespace.sweeps`.

Caching
-------
Identical requests return the cached :class:`SweepResponse` from a
small process-local LRU. The cache key is the SHA-256 of the
canonical-JSON request payload, so float / int order does not
matter. Cache size is small (32 entries) because a single user
session typically thrashes a few axes; restart the process to
flush.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from roverdevkit.tradespace.sweeps import (
    SWEEPABLE_VARIABLES,
    SweepAxis,
    SweepSpec,
    compute_sensitivity,
)
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_correction,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import SweepRequest, SweepResponse, SweepSensitivityOut
from webapp.backend.services.sweep import run_sweep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sweep"])


def _request_hash(req: SweepRequest) -> str:
    """SHA-256 of the request payload, after canonical JSON serialisation.

    Stable across Pydantic round-trips because ``model_dump_json``
    sorts keys via ``json.dumps(sort_keys=True)``.
    """
    payload = json.loads(req.model_dump_json())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=32)
def _cached_sweep(_request_key: str, req_json: str) -> SweepResponse:
    """LRU-cached sweep dispatch keyed on the request hash.

    The first argument is the cache key (the SHA-256 hex digest);
    keeping it as a parameter rather than computing it inside lets
    the LRU machinery hash a small string instead of re-walking the
    JSON for every cache lookup.

    The second argument is the canonical JSON of the request, parsed
    once inside this function and dispatched. Both are passed
    explicitly so the cache key + the inputs cannot drift.
    """
    req = SweepRequest.model_validate_json(req_json)

    scenarios = get_canonical_scenarios()
    if req.scenario_name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown scenario {req.scenario_name!r}. "
                f"Pick one of {sorted(scenarios.keys())}."
            ),
        )
    scenario = scenarios[req.scenario_name]
    if req.operational_duty_cycle is not None:
        # SCHEMA_VERSION v7_1: δ_ops is a true LHS-sampled surrogate
        # input, so applying the override here keeps both sweep
        # backends (surrogate batch predict + per-cell evaluator) in
        # sync with the Single-design tab's δ_ops slider. The grid
        # is still one-shot; only the *constant-across-grid* δ_ops
        # changes.
        scenario = scenario.model_copy(
            update={"operational_duty_cycle": req.operational_duty_cycle}
        )

    for ax in (req.x_axis, req.y_axis):
        if ax is None:
            continue
        if ax.variable not in SWEEPABLE_VARIABLES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"axis variable {ax.variable!r} is not sweepable. "
                    f"Allowed: {list(SWEEPABLE_VARIABLES)}."
                ),
            )

    spec = SweepSpec(
        target=req.target,
        x_axis=SweepAxis(
            variable=req.x_axis.variable,
            lo=req.x_axis.lo,
            hi=req.x_axis.hi,
            n_points=req.x_axis.n_points,
        ),
        y_axis=(
            None
            if req.y_axis is None
            else SweepAxis(
                variable=req.y_axis.variable,
                lo=req.y_axis.lo,
                hi=req.y_axis.hi,
                n_points=req.y_axis.n_points,
            )
        ),
        backend=req.backend,
    )

    soil = get_soil_for_simulant(scenario.soil_simulant)
    correction = get_correction()
    bundles = get_quantile_bundles()

    try:
        result = run_sweep(
            spec,
            req.base_design,
            scenario,
            soil,
            correction=correction,
            bundles=bundles,
        )
    except ValueError as exc:
        # Cell-count overflows + grid construction errors land here;
        # 422 is the right HTTP status for "input was structurally
        # valid but semantically over budget".
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    z_values: list[float] | list[list[float]]
    if result.y_values is None:
        z_values = [float(v) for v in result.z_values.tolist()]
    else:
        z_values = [[float(v) for v in row] for row in result.z_values.tolist()]

    used_scm_correction = result.backend_used == "evaluator" and correction is not None

    sens = compute_sensitivity(result)

    return SweepResponse(
        target=spec.target,
        scenario_name=req.scenario_name,
        x_variable=spec.x_axis.variable,
        y_variable=spec.y_axis.variable if spec.y_axis is not None else None,
        x_values=[float(v) for v in result.x_values.tolist()],
        y_values=(
            None
            if result.y_values is None
            else [float(v) for v in result.y_values.tolist()]
        ),
        z_values=z_values,
        backend_used=result.backend_used,  # type: ignore[arg-type]
        backend_requested=req.backend,
        used_scm_correction=used_scm_correction,
        n_cells=spec.n_cells(),
        elapsed_ms=result.elapsed_s * 1000.0,
        sensitivity=SweepSensitivityOut(
            total_spread=sens.total_spread,
            relative_spread=sens.relative_spread,
            axis_spread_x=sens.axis_spread_x,
            axis_spread_y=sens.axis_spread_y,
        ),
    )


@router.post("/sweep", response_model=SweepResponse)
def sweep_route(req: SweepRequest) -> SweepResponse:
    """1-D or 2-D parametric sweep over the design vector."""
    key = _request_hash(req)
    return _cached_sweep(key, req.model_dump_json())
