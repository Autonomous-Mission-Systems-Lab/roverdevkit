"""Validation endpoints (``/validate``) — currently the rediscovery view.

The rediscovery sub-route serves the precomputed Leave-One-Out
NSGA-II artifacts written by ``scripts/run_rediscovery_loo.py`` to
``reports/rediscovery_loo_evaluator/`` (corrected-evaluator-truth
fronts; the source of truth) and
``reports/rediscovery_loo_surrogate_v9/`` (surrogate-backed
fronts, offered as a wall-clock benchmark alongside the evaluator
result).

The frontend uses these to render the **headline paper figure**
for Phase-3: each registry rover's published design vector
overlaid on the optimiser's class-generic-scenario Pareto front,
labelled by flown-vs-design-target tier so reviewers can see at a
glance how the optimiser places each rover.

There is no live-compute path here on purpose. The evaluator-backed
sweep is ~11 minutes wall-clock; we serve the committed artifact
and document the regeneration recipe in
``reports/rediscovery_loo_comparison.md``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from roverdevkit.schema import DesignVector
from roverdevkit.validation.rover_registry import RoverRegistryEntry
from webapp.backend.loaders import (
    RediscoveryBackend,
    get_rediscovery_loo,
    get_registry,
)
from webapp.backend.schemas import (
    RediscoveryDetail,
    RediscoveryListResponse,
    RediscoveryParetoPoint,
    RediscoverySummary,
)

router = APIRouter(prefix="/validate", tags=["validate"])


def _is_flown_by_name(name: str) -> bool:
    """Look up ``RoverRegistryEntry.is_flown`` by ``rover_name``.

    Falls back to ``False`` for names not present in the registry
    (defensive — the rediscovery JSON's ``rover_name`` is
    authoritative and is written by ``rover_rediscovery.py`` from
    the same registry, so a mismatch would only happen mid-edit).
    """
    for entry in get_registry():
        if entry.rover_name == name:
            return entry.is_flown
    return False


def _summary_from_payload(
    slug: str,
    payload: dict[str, Any],
    backend: RediscoveryBackend,
) -> RediscoverySummary:
    return RediscoverySummary(
        slug=slug,
        rover_name=payload["rover_name"],
        is_flown=_is_flown_by_name(payload["rover_name"]),
        class_generic_scenario=payload["class_generic_scenario"],
        backend=backend,
        design_space_distance=float(payload["design_space_distance"]),
        pareto_dominated=bool(payload["pareto_dominated"]),
        mass_budget_kg=float(payload["mass_budget_kg"]),
        pareto_front_size=len(payload.get("pareto_front", [])),
    )


def _detail_from_payload(
    slug: str,
    payload: dict[str, Any],
    backend: RediscoveryBackend,
) -> RediscoveryDetail:
    # The committed JSONs persist the optimiser's nearest-Pareto pick
    # plus the rover's metrics under the generic scenario; the rover's
    # *published* design vector lives on the registry. Pull it from
    # there so the overlay marker uses the citation-grade design
    # rather than the optimiser's nearest neighbour.
    rover_design = _registry_design(payload["rover_name"])
    nearest_design = DesignVector.model_validate(payload["nearest_pareto_design"])
    front = [
        RediscoveryParetoPoint(
            design=DesignVector.model_validate(p["design"]),
            metrics={k: float(v) for k, v in p["metrics"].items()},
        )
        for p in payload.get("pareto_front", [])
    ]
    return RediscoveryDetail(
        slug=slug,
        rover_name=payload["rover_name"],
        is_flown=_is_flown_by_name(payload["rover_name"]),
        class_generic_scenario=payload["class_generic_scenario"],
        backend=backend,
        rover_design=rover_design,
        rover_metrics_under_generic_scenario={
            k: float(v) for k, v in payload["rover_metrics_under_generic_scenario"].items()
        },
        design_space_distance=float(payload["design_space_distance"]),
        pareto_dominated=bool(payload["pareto_dominated"]),
        mass_budget_kg=float(payload["mass_budget_kg"]),
        integer_matches={k: bool(v) for k, v in payload.get("integer_matches", {}).items()},
        per_variable_percent_errors={
            k: float(v) for k, v in payload.get("per_variable_percent_errors", {}).items()
        },
        nearest_pareto_index=int(payload["nearest_pareto_index"]),
        nearest_pareto_design=nearest_design,
        nearest_pareto_metrics={
            k: float(v) for k, v in payload["nearest_pareto_metrics"].items()
        },
        pareto_front=front,
    )


def _registry_design(rover_name: str) -> DesignVector:
    """Look up the rover's published ``DesignVector`` by name.

    Raises 404 inside the route if the rediscovery JSON references
    a rover that's no longer in the registry — that's a structural
    mismatch (someone edited one without the other) and is worth
    surfacing rather than silently masking.
    """
    for entry in get_registry():
        if entry.rover_name == rover_name:
            return entry.design
    raise HTTPException(
        status_code=404,
        detail=(
            f"rover {rover_name!r} appears in the rediscovery JSON but is "
            "no longer in the registry. Re-run scripts/run_rediscovery_loo.py "
            "to regenerate consistent artifacts."
        ),
    )


@router.get("/rediscovery", response_model=RediscoveryListResponse)
def list_rediscovery(
    backend: RediscoveryBackend = Query(
        default="evaluator",
        description=(
            "Which precomputed sweep to serve. ``evaluator`` "
            "(default) is the corrected-evaluator-truth sweep "
            "from reports/rediscovery_loo_evaluator/; "
            "``surrogate`` is the v9 surrogate-backed sweep "
            "from reports/rediscovery_loo_surrogate_v9/."
        ),
    ),
) -> RediscoveryListResponse:
    """List every rover's rediscovery summary for one backend.

    Returns 404 if the backend's directory is missing entirely
    (e.g. a fresh clone hasn't run the surrogate sweep yet); a
    half-empty list would be misleading because reviewers compare
    counts across backends.
    """
    try:
        artifacts = get_rediscovery_loo(backend)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    rovers = [_summary_from_payload(slug, payload, backend) for slug, payload in artifacts.items()]
    rovers.sort(key=lambda r: (not r.is_flown, r.rover_name))
    return RediscoveryListResponse(backend=backend, rovers=rovers)


@router.get("/rediscovery/{slug}", response_model=RediscoveryDetail)
def get_rediscovery_detail(
    slug: str,
    backend: RediscoveryBackend = Query(default="evaluator"),
) -> RediscoveryDetail:
    """Return the full rediscovery payload for one rover under one backend.

    The payload includes the rover's published design vector (for
    the overlay marker), its metrics under the class-generic
    scenario, the full Pareto front, and the design-space-distance
    / Pareto-dominance verdicts — everything the rediscovery page
    needs in a single request.
    """
    try:
        artifacts = get_rediscovery_loo(backend)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if slug not in artifacts:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown rediscovery slug {slug!r}. Available: "
                f"{sorted(artifacts.keys())}"
            ),
        )
    return _detail_from_payload(slug, artifacts[slug], backend)
