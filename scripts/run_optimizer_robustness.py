"""Assess NSGA-II repeatability and budget convergence for paper Pareto fronts.

The canonical fronts in ``reports/pareto_fronts`` are deliberately small enough
to regenerate on a laptop.  This script runs the same evaluator-backed pipeline
across several seeds and generation budgets, then writes summary artifacts that
support the manuscript's optimizer-robustness claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import (
    DEFAULT_OBJECTIVES,
    DESIGN_BOUNDS,
    NSGA2Runner,
    OptimizationConstraint,
)
from roverdevkit.validation.rover_rediscovery import _scenario_panel_orientation

from scripts.generate_pareto_fronts import (
    DEFAULT_EVALUATOR_EVAL_CAP,
    DEFAULT_RANGE_FLOOR_KM,
    SCENARIO_OVERRIDES,
)

DEFAULT_SEEDS = (12, 112)
DEFAULT_GENERATIONS = (30, 60, 90)
MASS_NORM_MAX_KG = 80.0
SLOPE_NORM_MAX_DEG = 45.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "optimizer_robustness",
        help="Directory for optimizer-robustness CSV/JSON/Markdown artifacts.",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to run. Defaults to all canonical scenarios.",
    )
    p.add_argument("--population-size", type=int, default=50)
    p.add_argument(
        "--generations",
        nargs="+",
        type=int,
        default=list(DEFAULT_GENERATIONS),
        help="Generation budgets to test.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds to run at each generation budget.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    fronts_dir = out_dir / "fronts"
    fronts_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_names(args.scenarios)
    range_floor = OptimizationConstraint(
        target="range_km", sense="min", value=DEFAULT_RANGE_FLOOR_KM
    )

    rows: list[dict[str, Any]] = []
    for scenario_name in scenarios:
        scenario = load_scenario(scenario_name)
        override = SCENARIO_OVERRIDES.get(scenario_name)
        objectives = DEFAULT_OBJECTIVES if override is None else override.objectives
        constraints = (
            (range_floor,)
            if override is None
            else (range_floor, *override.extra_constraints)
        )
        if override is not None and override.traverse_distance_m is not None:
            scenario = scenario.model_copy(
                update={"traverse_distance_m": override.traverse_distance_m}
            )

        soil = get_soil_parameters(scenario.soil_simulant)
        panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
        max_generations = max(args.generations)
        max_budget_fronts: list[pd.DataFrame] = []

        for generations in args.generations:
            for seed in args.seeds:
                t0 = time.perf_counter()
                result = NSGA2Runner(
                    scenario,
                    soil,
                    backend="evaluator",
                    objectives=objectives,
                    constraints=constraints,
                    population_size=args.population_size,
                    n_generations=generations,
                    seed=seed,
                    evaluator_eval_cap=DEFAULT_EVALUATOR_EVAL_CAP,
                    panel_tilt_deg=panel_tilt_deg,
                    panel_azimuth_deg=panel_azimuth_deg,
                ).run()
                elapsed_s = time.perf_counter() - t0
                front = result.to_frame()
                front.insert(0, "seed", seed)
                front.insert(0, "generations", generations)
                front.insert(0, "scenario_name", scenario_name)
                front_path = (
                    fronts_dir / f"front_{scenario_name}_g{generations}_s{seed}.csv"
                )
                front.to_csv(front_path, index=False)
                if generations == max_generations:
                    max_budget_fronts.append(front)

                row = _summarize_front(
                    front,
                    scenario_name=scenario_name,
                    generations=generations,
                    seed=seed,
                    elapsed_s=elapsed_s,
                    traverse_distance_km=scenario.traverse_distance_m / 1000.0,
                    objectives=objectives,
                    front_csv=front_path,
                )
                rows.append(row)
                print(
                    f"{scenario_name} g={generations} seed={seed}: "
                    f"hv={row['normalized_hypervolume']:.3f}, "
                    f"n={row['pareto_size']} ({elapsed_s:.1f} s)",
                    flush=True,
                )

        reference = (
            pd.concat(max_budget_fronts, ignore_index=True)
            if max_budget_fronts
            else pd.DataFrame()
        )
        for row in rows:
            if row["scenario_name"] != scenario_name:
                continue
            front = pd.read_csv(row["front_csv"])
            row["median_distance_to_max_budget_front"] = _median_distance_to_reference(
                front,
                reference,
                traverse_distance_km=scenario.traverse_distance_m / 1000.0,
                objectives=objectives,
            )

    per_run = pd.DataFrame(rows)
    per_run_path = out_dir / "optimizer_robustness_runs.csv"
    per_run.to_csv(per_run_path, index=False)

    summary = _aggregate(per_run)
    summary_path = out_dir / "optimizer_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "population_size": args.population_size,
        "generations": args.generations,
        "seeds": args.seeds,
        "scenarios": scenarios,
        "per_run_csv": str(per_run_path),
        "summary_csv": str(summary_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_markdown(out_dir / "optimizer_robustness_report.md", summary, manifest)
    return 0


def _scenario_names(raw: list[str] | None) -> list[str]:
    allowed = set(list_scenarios())
    values = list_scenarios() if raw is None else raw
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; allowed: {sorted(allowed)}")
    return list(values)


def _summarize_front(
    front: pd.DataFrame,
    *,
    scenario_name: str,
    generations: int,
    seed: int,
    elapsed_s: float,
    traverse_distance_km: float,
    objectives: tuple[Any, ...],
    front_csv: Path,
) -> dict[str, Any]:
    if front.empty:
        return {
            "scenario_name": scenario_name,
            "generations": generations,
            "seed": seed,
            "elapsed_s": elapsed_s,
            "pareto_size": 0,
            "normalized_hypervolume": 0.0,
            "front_csv": str(front_csv),
        }

    return {
        "scenario_name": scenario_name,
        "generations": generations,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "pareto_size": len(front),
        "normalized_hypervolume": _normalized_hypervolume(
            front,
            traverse_distance_km=traverse_distance_km,
            objectives=objectives,
        ),
        "max_range_km": float(front["range_km"].max()),
        "min_mass_kg": float(front["total_mass_kg"].min()),
        "max_slope_capability_deg": float(front["slope_capability_deg"].max()),
        "four_wheel_pct": float((front["n_wheels"] == 4).mean() * 100.0),
        "width_floor_pct": float(
            (front["wheel_width_m"] <= DESIGN_BOUNDS["wheel_width_m"][0] + 1e-3).mean()
            * 100.0
        ),
        "radius_ceiling_pct": float(
            (front["wheel_radius_m"] >= DESIGN_BOUNDS["wheel_radius_m"][1] - 1e-3).mean()
            * 100.0
        ),
        "grouser_ceiling_pct": float(
            (
                front["grouser_height_m"]
                >= DESIGN_BOUNDS["grouser_height_m"][1] - 1e-4
            ).mean()
            * 100.0
        ),
        "front_csv": str(front_csv),
    }


def _aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "normalized_hypervolume",
        "median_distance_to_max_budget_front",
        "max_range_km",
        "min_mass_kg",
        "max_slope_capability_deg",
        "four_wheel_pct",
        "width_floor_pct",
        "radius_ceiling_pct",
        "grouser_ceiling_pct",
    ]
    grouped = per_run.groupby(["scenario_name", "generations"], sort=True)
    out = grouped[numeric].agg(["mean", "std", "min", "max"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in col if part)
        for col in out.columns.to_flat_index()
    ]
    out["n_runs"] = grouped.size().to_numpy()
    return out


def _normalized_hypervolume(
    front: pd.DataFrame,
    *,
    traverse_distance_km: float,
    objectives: tuple[Any, ...],
) -> float:
    F = _normalized_objectives(front, traverse_distance_km, objectives)
    if F.size == 0:
        return 0.0
    return float(HV(ref_point=np.ones(F.shape[1]) * 1.05).do(F))


def _median_distance_to_reference(
    front: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    traverse_distance_km: float,
    objectives: tuple[Any, ...],
) -> float:
    if front.empty or reference.empty:
        return float("nan")
    F = _normalized_objectives(front, traverse_distance_km, objectives)
    R = _normalized_objectives(reference, traverse_distance_km, objectives)
    distances = np.sqrt(((F[:, None, :] - R[None, :, :]) ** 2).sum(axis=2))
    return float(np.median(np.min(distances, axis=1)))


def _normalized_objectives(
    front: pd.DataFrame,
    traverse_distance_km: float,
    objectives: tuple[Any, ...],
) -> np.ndarray:
    values: list[np.ndarray] = []
    for objective in objectives:
        target = objective.target
        raw = front[target].to_numpy(dtype=float)
        if target == "range_km":
            norm = raw / max(traverse_distance_km, 1e-9)
        elif target == "total_mass_kg":
            norm = raw / MASS_NORM_MAX_KG
        elif target == "slope_capability_deg":
            norm = raw / SLOPE_NORM_MAX_DEG
        else:
            raise ValueError(f"unsupported objective target {target!r}")

        norm = np.clip(norm, 0.0, 1.0)
        values.append(norm if objective.direction == "min" else 1.0 - norm)
    return np.column_stack(values)


def _write_markdown(path: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    cols = [
        "scenario_name",
        "generations",
        "n_runs",
        "normalized_hypervolume_mean",
        "normalized_hypervolume_std",
        "median_distance_to_max_budget_front_mean",
        "four_wheel_pct_mean",
        "width_floor_pct_mean",
    ]
    lines = [
        "# Optimizer Robustness",
        "",
        (
            f"Population size {manifest['population_size']}; seeds "
            f"{manifest['seeds']}; generation budgets {manifest['generations']}."
        ),
        "",
        _markdown_table(summary[cols]),
        "",
        "Per-run CSV: `optimizer_robustness_runs.csv`.",
        "Summary CSV: `optimizer_robustness_summary.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
