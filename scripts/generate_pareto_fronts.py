"""Generate canonical evaluator-driven Pareto fronts for the webapp and paper.

The Pareto Explorer ships one precomputed front per canonical scenario so
that a fresh clone gets a working visualization without running NSGA-II
live. The canonical fronts are produced with the **corrected analytical
+ SCM-correction evaluator** as the fitness function: every Pareto point
is therefore evaluator-truth, not a surrogate prediction.

The surrogate stays the default for the *live* Optimize tab inside the
webapp (where ~1 ms / call lets NSGA-II finish in seconds), but the
canonical artifacts that ship with the repo and drive the paper figures
come from the physics evaluator. See ``project/README`` and the
"Reproducibility" section there for the broader rationale.

Outputs under ``--out-dir`` (defaults to ``reports/pareto_fronts``):

- ``front_<scenario>.csv``      — one Pareto point per row, design
  fields + the four primary metrics + ``backend_used = "evaluator"``.
- ``front_<scenario>.metadata.json`` — population/generations/seed,
  objectives, evaluator cost, and the correction artifact's path.
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
from roverdevkit.terramechanics.correction_model import (
    DEFAULT_CORRECTION_PATH,
    WheelLevelCorrection,
    load_correction_or_none,
)
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
# corrected evaluator (~22 ms / call) finishes one scenario in ~1 min
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
    p.add_argument(
        "--no-correction",
        action="store_true",
        help=(
            "Run the analytical-only (Bekker-Wong) evaluator instead of the "
            "corrected BW + SCM evaluator. The shipped artifacts use the "
            "correction; this flag is for environments without the SCM "
            "correction joblib."
        ),
    )
    p.add_argument(
        "--correction-path",
        type=Path,
        default=None,
        help=(
            "Override path to the wheel-level correction joblib. Defaults to "
            "the package default (data/scm/correction_v1.joblib)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    correction, correction_path = _load_correction(args)
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
            correction=correction,
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
            "correction_path": _repo_relative(correction_path) if correction_path else None,
            "correction_enabled": correction is not None,
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


def _load_correction(args: argparse.Namespace) -> tuple[WheelLevelCorrection | None, Path | None]:
    if args.no_correction:
        return None, None
    path = (args.correction_path or DEFAULT_CORRECTION_PATH).expanduser().resolve()
    correction = load_correction_or_none(path, on_missing="warn")
    return correction, path


def _repo_relative(path: Path) -> str:
    """Return ``path`` relative to the repo root when possible, else as-is.

    Keeps the checked-in metadata portable across machines without
    leaking user-specific absolute paths into git.
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _scenario_names(raw: list[str] | None) -> list[ScenarioName]:
    allowed = set(list_scenarios())
    values = list_scenarios() if raw is None else raw
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; allowed: {sorted(allowed)}")
    return [name for name in values]  # type: ignore[list-item]


if __name__ == "__main__":
    sys.exit(main())
