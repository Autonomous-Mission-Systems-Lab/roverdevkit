"""Run the Layer-5 rediscovery LOO sweep and write the paper artifacts.

Drives
:func:`roverdevkit.validation.rediscovery_report.run_rediscovery_loo`
over every registry rover (flown by default; pass ``--all`` to include
design-target rovers too) and emits the artifact set documented in
:func:`roverdevkit.validation.rediscovery_report.write_loo_artifacts`.

Per-rover NSGA-II hyperparameters and ``mass_ceiling_slop`` defaults
live in :data:`DEFAULT_PER_ROVER_OVERRIDES`; passing
``--no-per-rover-overrides`` disables them (useful for sensitivity
analysis - expect CADRE-unit to fail under uniform defaults).

Outputs (under ``--out-dir``, defaults to ``reports/rediscovery_loo``):

- ``summary.csv`` — one-row-per-rover summary table
- ``<rover>.json`` — per-rover full Pareto front and scoring detail
- ``failures.json`` — ``{rover_name: error_message}`` (empty if none)
- ``rediscovery_loo_report.md`` — human-readable rollup

Usage
-----
::

    python scripts/run_rediscovery_loo.py
    python scripts/run_rediscovery_loo.py --all --seed 42
    python scripts/run_rediscovery_loo.py --out-dir reports/rediscovery_loo_paper
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib

from roverdevkit.validation.rediscovery_report import (
    DEFAULT_PER_ROVER_OVERRIDES,
    run_rediscovery_loo,
    write_loo_artifacts,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rediscovery_loo"),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Include design-target rovers (MoonRanger, Rashid-1, Tenacious, "
            "CADRE-unit) in addition to the flown rovers (Pragyan, Yutu-2). "
            "Default: flown only."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population-size", type=int, default=60)
    p.add_argument("--n-generations", type=int, default=16)
    p.add_argument("--mass-ceiling-slop", type=float, default=0.10)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help=(
            "Number of NSGA-II seeds to ensemble per rover. 1 (default) "
            "preserves single-seed historical behaviour; 5 is a paper-"
            "grade ensemble (~5x evaluator budget per rover)."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("evaluator", "surrogate"),
        default="evaluator",
        help=(
            "evaluator = corrected physics (default, ~20 ms/design); "
            "surrogate = calibrated quantile-XGB heads (~0.1 ms/design)."
        ),
    )
    p.add_argument(
        "--quantile-bundles",
        type=Path,
        default=Path("reports/surrogate_v9/quantile_bundles.joblib"),
        help="Path to quantile bundles joblib; required when --backend=surrogate.",
    )
    p.add_argument(
        "--evaluator-eval-cap",
        type=int,
        default=1000,
        help=(
            "Evaluator-backend NSGA-II evaluation cap per seed. Default 1000 "
            "matches the webapp safety cap; paper-grade runs typically use "
            "10_000-25_000."
        ),
    )
    p.add_argument(
        "--no-per-rover-overrides",
        action="store_true",
        help=(
            "Disable DEFAULT_PER_ROVER_OVERRIDES (every rover uses the "
            "uniform default budget). Expect CADRE-unit to land in "
            "failures.json under this mode."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    bundles = None
    if args.backend == "surrogate":
        if not args.quantile_bundles.exists():
            raise FileNotFoundError(
                f"--backend=surrogate requires quantile bundles; "
                f"file not found: {args.quantile_bundles}"
            )
        bundles = joblib.load(args.quantile_bundles)

    overrides = {} if args.no_per_rover_overrides else dict(DEFAULT_PER_ROVER_OVERRIDES)
    summary = run_rediscovery_loo(
        flown_only=not args.all,
        seed=args.seed,
        default_population_size=args.population_size,
        default_n_generations=args.n_generations,
        default_mass_ceiling_slop=args.mass_ceiling_slop,
        per_rover_overrides=overrides,
        n_seeds=args.n_seeds,
        backend=args.backend,
        bundles=bundles,
        evaluator_eval_cap=args.evaluator_eval_cap,
    )
    written = write_loo_artifacts(summary, args.out_dir)

    print(f"Wrote {len(written)} artifact(s) to {args.out_dir}:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")
    print()
    print(f"Rovers succeeded: {len(summary.results)}")
    print(f"Rovers failed:    {len(summary.failures)}")
    for rover_name, msg in summary.failures.items():
        print(f"  - {rover_name}: {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
