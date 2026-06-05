"""Generate canonical evaluator-driven Pareto fronts for the webapp and paper.

The Pareto Explorer ships one precomputed front per canonical scenario so
that a fresh clone gets a working visualization without running NSGA-II
live. The canonical fronts are produced with the **analytical Bekker-Wong
evaluator** as the fitness function: every Pareto point is therefore
evaluator-truth, not a surrogate prediction.

The surrogate stays the default for the *live* Optimize tab inside the
webapp (where ~1 ms / call lets NSGA-II finish in seconds), but the
canonical artifacts that ship with the repo and drive the paper figures
come from the physics evaluator. See ``project/README`` and the
"Reproducibility" section there for the broader rationale.

Outputs under ``--out-dir`` (defaults to ``reports/pareto_fronts``):

- ``front_<scenario>.csv``      — one Pareto point per row, design
  fields + the four primary metrics + ``backend_used = "evaluator"``.
- ``front_<scenario>.metadata.json`` — population/generations/seed,
  objectives, and evaluator cost.
- ``manifest.json`` — aggregate metadata across scenarios.

Example
-------
::

    conda run -n roverdevkit --no-capture-output \\
        python scripts/generate_pareto_fronts.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import ScenarioName
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import (
    DEFAULT_OBJECTIVES,
    NSGA2Runner,
    OptimizationConstraint,
)
# Single source of truth for the fixed-tilt panel approximation so the
# canonical fronts use the *same* polar insolation physics as the
# leakage-controlled rediscovery sweep. Without it the high-latitude
# scenarios (polar_prospecting at lat=-85) evaluate every candidate
# with a horizontal panel (~18x insolation deficit) and, once the v9
# scientific-payload power requirement is added, NSGA-II returns an
# empty front because no design clears the range floor.
from roverdevkit.validation.rover_rediscovery import _scenario_panel_orientation

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default NSGA-II budget for the canonical offline run. Sized so the
# analytical evaluator (~22 ms / call) finishes one scenario in ~1 min
# and all four scenarios in ~4 min on a laptop. The Pareto Explorer
# only needs a few dozen non-dominated points to communicate the
# tradeoff shape, so denser fronts would buy little but cost more.
DEFAULT_POPULATION_SIZE = 50
DEFAULT_GENERATIONS = 60

# Generous cap so the offline script can run higher budgets when a
# reviewer asks for a denser front. The live webapp Optimize route
# keeps the constructor's default 1000-eval cap.
DEFAULT_EVALUATOR_EVAL_CAP = 50_000

# Stalled designs report range_km == 0 but can still win on
# slope_capability_deg (slope is a static-load check) or total_mass_kg
# (light = low mass). Without a floor on range they pollute the
# Pareto front with non-navigable rovers. 0.1 km is permissive — it
# only filters the binary stall failure, not slow-but-feasible
# designs — and matches the live Optimize tab's default constraint.
DEFAULT_RANGE_FLOOR_KM = 0.1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "pareto_fronts",
        help="Directory for front_<scenario>.csv and metadata JSON files.",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to generate. Defaults to all canonical scenarios.",
    )
    p.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
    )
    p.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_names(args.scenarios)

    constraints = (
        OptimizationConstraint(target="range_km", sense="min", value=DEFAULT_RANGE_FLOOR_KM),
    )

    manifest: list[dict[str, Any]] = []
    for i, scenario_name in enumerate(scenarios):
        scenario = load_scenario(scenario_name)
        soil = get_soil_parameters(scenario.soil_simulant)
        panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
        seed = args.seed + i
        t0 = time.perf_counter()
        result = NSGA2Runner(
            scenario,
            soil,
            backend="evaluator",
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
        front.insert(0, "scenario_name", scenario_name)
        front_path = out_dir / f"front_{scenario_name}.csv"
        front.to_csv(front_path, index=False)

        metadata = {
            "scenario_name": scenario_name,
            "backend": result.backend_used,
            "dataset_version": os.environ.get("ROVERDEVKIT_DATASET_VERSION", "v9"),
            "objectives": [
                {"target": obj.target, "direction": obj.direction}
                for obj in DEFAULT_OBJECTIVES
            ],
            "constraints": [
                {"target": c.target, "sense": c.sense, "value": c.value}
                for c in constraints
            ],
            "population_size": args.population_size,
            "generations": args.generations,
            "seed": seed,
            "panel_tilt_deg": panel_tilt_deg,
            "panel_azimuth_deg": panel_azimuth_deg,
            "elapsed_s": elapsed_s,
            "pareto_size": len(result.design_vectors),
            "front_csv": str(front_path),
        }
        meta_path = out_dir / f"front_{scenario_name}.metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(metadata)
        print(
            f"{scenario_name}: wrote {len(front)} points to {front_path} "
            f"({elapsed_s:.1f} s, evaluator)",
            flush=True,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest {manifest_path}", flush=True)
    return 0


def _scenario_names(raw: list[str] | None) -> list[ScenarioName]:
    allowed = set(list_scenarios())
    values = list_scenarios() if raw is None else raw
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; allowed: {sorted(allowed)}")
    return [name for name in values]  # type: ignore[list-item]


if __name__ == "__main__":
    sys.exit(main())
