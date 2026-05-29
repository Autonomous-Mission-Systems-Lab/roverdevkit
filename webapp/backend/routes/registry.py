"""Real-rover registry endpoints (``/registry``)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from roverdevkit.validation.rover_registry import RoverRegistryEntry
from webapp.backend.loaders import get_registry
from webapp.backend.schemas import RegistryEntrySummary, RegistryListResponse

router = APIRouter(prefix="/registry", tags=["registry"])


def _to_summary(entry: RoverRegistryEntry) -> RegistryEntrySummary:
    return RegistryEntrySummary(
        rover_name=entry.rover_name,
        is_flown=entry.is_flown,
        design=entry.design,
        scenario=entry.scenario,
        gravity_m_per_s2=entry.gravity_m_per_s2,
        thermal_architecture=asdict(entry.thermal_architecture),
        panel_efficiency=entry.panel_efficiency,
        panel_dust_factor=entry.panel_dust_factor,
        panel_tilt_deg=entry.panel_tilt_deg,
        panel_azimuth_deg=entry.panel_azimuth_deg,
        imputation_notes=entry.imputation_notes,
    )


@router.get("", response_model=RegistryListResponse)
def list_registry() -> RegistryListResponse:
    """Return all registry entries (flown + design-target)."""
    return RegistryListResponse(rovers=[_to_summary(e) for e in get_registry()])


@router.get("/{name}", response_model=RegistryEntrySummary)
def get_rover(name: str) -> RegistryEntrySummary:
    """Single registry entry by ``rover_name`` (case-sensitive)."""
    for entry in get_registry():
        if entry.rover_name == name:
            return _to_summary(entry)
    available = [e.rover_name for e in get_registry()]
    raise HTTPException(
        status_code=404,
        detail=f"unknown rover {name!r}. Available: {available}",
    )
