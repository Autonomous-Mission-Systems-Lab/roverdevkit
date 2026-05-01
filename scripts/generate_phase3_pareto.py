"""Generate canonical Phase-3 Pareto fronts for the webapp and paper figures.

The Week-12 explorer should not depend on a reviewer running a live
NSGA-II job. This script precomputes one surrogate-backed front for
each canonical scenario and writes CSV + metadata JSON artifacts under
``reports/phase3_pareto/``.

Example
-------
::

    conda run -n roverdevkit python scripts/generate_phase3_pareto.py \\
        --population-size 96 \\
        --generations 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import ScenarioName
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import DEFAULT_OBJECTIVES, NSGA2Runner

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = REPO_ROOT / "reports" / "week12_intervals_v7_1" / "quantile_bundles.joblib"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "phase3_pareto",
        help="Directory for front_<scenario>.csv and metadata JSON files.",
    )
    p.add_argument(
        "--quantile-bundles",
        type=Path,
        default=None,
        help="Override path to quantile_bundles.joblib. Defaults to backend config.",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario names to generate. Defaults to all canonical scenarios.",
    )
    p.add_argument("--population-size", type=int, default=96)
    p.add_argument("--generations", type=int, default=80)
    p.add_argument("--seed", type=int, default=12)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle_path = (
        args.quantile_bundles.expanduser().resolve()
        if args.quantile_bundles is not None
        else Path(os.environ.get("ROVERDEVKIT_QUANTILE_BUNDLES", DEFAULT_BUNDLE_PATH)).resolve()
    )
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"quantile bundles not found at {bundle_path}. "
            "Run scripts/calibrate_intervals.py or pass --quantile-bundles."
        )

    bundles = joblib.load(bundle_path)
    scenarios = _scenario_names(args.scenarios)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for i, scenario_name in enumerate(scenarios):
        scenario = load_scenario(scenario_name)
        soil = get_soil_parameters(scenario.soil_simulant)
        seed = args.seed + i
        t0 = time.perf_counter()
        result = NSGA2Runner(
            scenario,
            soil,
            bundles=bundles,
            population_size=args.population_size,
            n_generations=args.generations,
            seed=seed,
        ).run()
        elapsed_s = time.perf_counter() - t0

        front = result.to_frame()
        front.insert(0, "scenario_name", scenario_name)
        front_path = out_dir / f"front_{scenario_name}.csv"
        front.to_csv(front_path, index=False)

        metadata = {
            "scenario_name": scenario_name,
            "backend": result.backend_used,
            "quantile_bundles_path": str(bundle_path),
            "dataset_version": os.environ.get("ROVERDEVKIT_DATASET_VERSION", "v7_1"),
            "objectives": [
                {"target": obj.target, "direction": obj.direction}
                for obj in DEFAULT_OBJECTIVES
            ],
            "constraints": [],
            "population_size": args.population_size,
            "generations": args.generations,
            "seed": seed,
            "elapsed_s": elapsed_s,
            "pareto_size": len(result.design_vectors),
            "front_csv": str(front_path),
        }
        meta_path = out_dir / f"front_{scenario_name}.metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(metadata)
        print(
            f"{scenario_name}: wrote {len(front)} points to {front_path} "
            f"({elapsed_s:.2f} s)",
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
