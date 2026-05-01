"""Persisted canonical Pareto-front routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from roverdevkit.schema import DesignVector
from webapp.backend.config import get_settings
from webapp.backend.schemas import (
    OptimizeParetoPoint,
    ParetoFrontListResponse,
    ParetoFrontResponse,
    ParetoFrontSummary,
)

router = APIRouter(tags=["pareto"])

PRIMARY_TARGETS = (
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
)


@router.get("/pareto/fronts", response_model=ParetoFrontListResponse)
def list_pareto_fronts() -> ParetoFrontListResponse:
    """List canonical fronts written by ``scripts/generate_phase3_pareto.py``."""
    out_dir = _front_dir()
    manifest = _read_manifest(out_dir)
    fronts = [
        ParetoFrontSummary(
            scenario_name=str(item["scenario_name"]),
            pareto_size=int(item.get("pareto_size", 0)),
            backend=str(item.get("backend", "unknown")),
            dataset_version=(
                str(item["dataset_version"]) if item.get("dataset_version") is not None else None
            ),
            front_url=f"/pareto/fronts/{item['scenario_name']}",
        )
        for item in manifest
    ]
    return ParetoFrontListResponse(fronts=fronts)


@router.get("/pareto/fronts/{scenario_name}", response_model=ParetoFrontResponse)
def get_pareto_front(scenario_name: str) -> ParetoFrontResponse:
    """Load one persisted canonical front."""
    out_dir = _front_dir()
    meta_path = out_dir / f"front_{scenario_name}.metadata.json"
    csv_path = out_dir / f"front_{scenario_name}.csv"
    if not meta_path.exists() or not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"canonical Pareto front {scenario_name!r} not found. "
                "Run scripts/generate_phase3_pareto.py first."
            ),
        )
    metadata = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path)
    points = [_row_to_point(row) for row in df.to_dict(orient="records")]
    return ParetoFrontResponse(
        scenario_name=scenario_name,
        metadata=metadata,
        pareto_front=points,
    )


def _front_dir() -> Path:
    return get_settings().repo_root / "reports" / "phase3_pareto"


def _read_manifest(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "manifest.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise HTTPException(status_code=500, detail=f"invalid Pareto manifest at {path}")
    return [item for item in raw if isinstance(item, dict)]


def _row_to_point(row: dict[str, Any]) -> OptimizeParetoPoint:
    design = DesignVector(
        wheel_radius_m=float(row["wheel_radius_m"]),
        wheel_width_m=float(row["wheel_width_m"]),
        grouser_height_m=float(row["grouser_height_m"]),
        grouser_count=int(row["grouser_count"]),
        n_wheels=int(row["n_wheels"]),
        chassis_mass_kg=float(row["chassis_mass_kg"]),
        wheelbase_m=float(row["wheelbase_m"]),
        solar_area_m2=float(row["solar_area_m2"]),
        battery_capacity_wh=float(row["battery_capacity_wh"]),
        avionics_power_w=float(row["avionics_power_w"]),
        peak_wheel_torque_nm=float(row["peak_wheel_torque_nm"]),
    )
    metrics = {target: float(row[target]) for target in PRIMARY_TARGETS}
    return OptimizeParetoPoint(design=design, metrics=metrics)
