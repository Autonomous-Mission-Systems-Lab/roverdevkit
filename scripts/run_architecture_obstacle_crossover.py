"""Sweep required obstacle height and locate the rocker-bogie crossover.

Re-runs the canonical evaluator-backed NSGA-II pipeline at increasing
``MissionScenario.required_obstacle_height_m`` values and records when
six-wheel rocker-bogie architectures enter the Pareto set.

Outputs under ``reports/architecture_obstacle_crossover/``:

- ``front_<scenario>__hobs_<value>.csv``
- ``crossover_summary.csv``
- ``manifest.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from roverdevkit.mission.scenarios import list_scenarios, load_scenario  # noqa: E402
from roverdevkit.schema import ScenarioName  # noqa: E402
from roverdevkit.terramechanics.soils import get_soil_parameters  # noqa: E402
from roverdevkit.tradespace.optimizer import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    NSGA2Runner,
    OptimizationConstraint,
)
from roverdevkit.validation.rover_rediscovery import _scenario_panel_orientation  # noqa: E402
from generate_pareto_fronts import (  # noqa: E402
    DEFAULT_EVALUATOR_EVAL_CAP,
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANGE_FLOOR_KM,
    SCENARIO_OVERRIDES,
)

DEFAULT_H_OBS_M: tuple[float, ...] = (
    0.0,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "architecture_obstacle_crossover",
    )
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument(
        "--h-obs-m",
        nargs="+",
        type=float,
        default=list(DEFAULT_H_OBS_M),
        help="Required obstacle heights to sweep (m).",
    )
    p.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    p.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    p.add_argument("--seed", type=int, default=12)
    return p.parse_args(argv)


def _scenario_names(raw: list[str] | None) -> list[ScenarioName]:
    allowed = set(list_scenarios())
    values = list_scenarios() if raw is None else raw
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; allowed: {sorted(allowed)}")
    return [name for name in values]  # type: ignore[list-item]


def _summary_row(
    scenario_name: str,
    h_obs_m: float,
    front: pd.DataFrame,
) -> dict[str, Any]:
    n = len(front)
    front_empty = n == 0 or "mobility_architecture" not in front.columns
    if front_empty:
        return {
            "scenario_name": scenario_name,
            "required_obstacle_height_m": h_obs_m,
            "n_points": n,
            "front_empty": True,
            "frac_rocker_bogie": float("nan"),
            "frac_rigid_4wheel": float("nan"),
            "min_mass_kg": float("nan"),
            "max_range_km": float("nan"),
            "median_obstacle_capability_m": float("nan"),
        }
    rocker = front["mobility_architecture"] == "rocker_bogie_6wheel"
    return {
        "scenario_name": scenario_name,
        "required_obstacle_height_m": h_obs_m,
        "n_points": n,
        "front_empty": False,
        "frac_rocker_bogie": float(rocker.mean()),
        "frac_rigid_4wheel": float((~rocker).mean()),
        "min_mass_kg": float(front["total_mass_kg"].min()),
        "max_range_km": float(front["range_km"].max()),
        "median_obstacle_capability_m": float(front["obstacle_capability_m"].median()),
    }


def _rocker_summary_label(row: dict[str, Any]) -> str:
    if row.get("front_empty"):
        return "empty"
    frac = row["frac_rocker_bogie"]
    if frac != frac:  # NaN
        return "n/a"
    return f"{frac:.0%}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_names(args.scenarios)
    h_values = [float(h) for h in args.h_obs_m]

    range_floor = OptimizationConstraint(
        target="range_km", sense="min", value=DEFAULT_RANGE_FLOOR_KM
    )
    obstacle_floor = OptimizationConstraint(
        target="obstacle_margin_m", sense="min", value=0.0
    )

    summary_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for i, scenario_name in enumerate(scenarios):
        override = SCENARIO_OVERRIDES.get(scenario_name)
        objectives = override.objectives if override else DEFAULT_OBJECTIVES
        extra = override.extra_constraints if override else ()
        panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(
            load_scenario(scenario_name)
        )

        for j, h_obs_m in enumerate(h_values):
            scenario = load_scenario(scenario_name).model_copy(
                update={"required_obstacle_height_m": h_obs_m}
            )
            if override is not None and override.traverse_distance_m is not None:
                scenario = scenario.model_copy(
                    update={"traverse_distance_m": override.traverse_distance_m}
                )

            soil = get_soil_parameters(scenario.soil_simulant)
            constraints: tuple[OptimizationConstraint, ...] = (
                range_floor,
                *extra,
            )
            if h_obs_m > 0.0:
                constraints = (*constraints, obstacle_floor)

            seed = args.seed + i * 100 + j
            t0 = time.perf_counter()
            result = NSGA2Runner(
                scenario,
                soil,
                backend="evaluator",
                objectives=objectives,
                constraints=constraints,
                population_size=args.population_size,
                n_generations=args.generations,
                seed=seed,
                evaluator_eval_cap=DEFAULT_EVALUATOR_EVAL_CAP,
                panel_tilt_deg=panel_tilt_deg,
                panel_azimuth_deg=panel_azimuth_deg,
            ).run()
            elapsed_s = time.perf_counter() - t0

            front = result.to_frame()
            tag = f"{h_obs_m:.3f}".replace(".", "p")
            front_path = out_dir / f"front_{scenario_name}__hobs_{tag}.csv"
            front.insert(0, "scenario_name", scenario_name)
            front.insert(1, "required_obstacle_height_m", h_obs_m)
            front.to_csv(front_path, index=False)

            summary_rows.append(_summary_row(scenario_name, h_obs_m, front))
            row = summary_rows[-1]
            manifest.append(
                {
                    "scenario_name": scenario_name,
                    "required_obstacle_height_m": h_obs_m,
                    "seed": seed,
                    "elapsed_s": elapsed_s,
                    "pareto_size": len(result.design_vectors),
                    "front_empty": row["front_empty"],
                    "front_csv": str(front_path),
                }
            )
            print(
                f"{scenario_name} h={h_obs_m:.3f} m: "
                f"rocker={_rocker_summary_label(row)} "
                f"({len(front)} pts, {elapsed_s:.1f}s)",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "crossover_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
