"""Generate a surrogate-training analytical dataset (LHS sampler -> evaluator -> Parquet).

Single canonical entry point for any dataset rebuild: pilot, full
training run, or release-time benchmark slice. The same flags drive
all three; see ``--help``.

Examples
--------
::

    # 200-sample smoke pilot (canonical reproduction)
    python scripts/build_dataset.py \\
        --n-per-scenario 50 \\
        --out data/analytical/lhs_pilot.parquet \\
        --seed 42 \\
        --workers 1 \\
        --notes "baseline-surrogate pilot pilot rebuild under v2."

    # Full 40k training set on the v3 widened bounds (current canonical)
    python scripts/build_dataset.py \\
        --n-per-scenario 10000 \\
        --out data/analytical/lhs_v3.parquet \\
        --seed 42 \\
        --notes "v3 widened LHS bounds (chassis 3-50 kg, wheel_width 0.03-0.20 m, grouser 0-0.020 m)."

The script writes a single Parquet file with the schema documented in
``data/analytical/SCHEMA.md`` (``SCHEMA_VERSION`` constant in
``roverdevkit.surrogate.dataset`` is the source of truth). Dataset-
level metadata (seed, n_per_scenario, fidelity, build timestamp,
free-form notes) is written to the file footer so re-runs are
reproducible from disk alone.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from roverdevkit.surrogate.dataset import (
    SCHEMA_VERSION,
    DatasetMetadata,
    build_and_write,
)
from roverdevkit.surrogate.sampling import FAMILIES, generate_samples

DEFAULT_FAMILIES: tuple[str, ...] = tuple(FAMILIES.keys())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-per-scenario",
        type=int,
        required=True,
        help="LHS samples per scenario family. Total rows = n * len(families).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output Parquet path (parent dirs are created).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampler RNG seed. Same seed -> same samples (default: 42).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Worker process count. 0 (default) => os.cpu_count() - 1 (capped at 1). "
            "1 => serial; useful for debugging or if multiprocessing/spawn misbehaves."
        ),
    )
    p.add_argument(
        "--families",
        nargs="+",
        choices=list(DEFAULT_FAMILIES),
        default=list(DEFAULT_FAMILIES),
        help="Scenario families to include (default: all four).",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1).",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Test split fraction (default: 0.1).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="multiprocessing.imap_unordered chunk size (default: 32).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes string written to the Parquet metadata.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for the build run (default: INFO).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("build_dataset")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    log.info(
        "schema=%s n_per_scenario=%d families=%d total_samples=%d workers=%d seed=%d "
        "out=%s",
        SCHEMA_VERSION,
        args.n_per_scenario,
        len(args.families),
        args.n_per_scenario * len(args.families),
        workers,
        args.seed,
        args.out,
    )

    log.info("generating LHS samples...")
    samples = generate_samples(
        n_per_scenario=args.n_per_scenario,
        seed=args.seed,
        scenario_names=list(args.families),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    log.info("generated %d samples", len(samples))

    meta = DatasetMetadata(
        sampler_seed=args.seed,
        n_per_scenario=args.n_per_scenario,
        scenario_families=tuple(args.families),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        notes=args.notes,
    )

    log.info("evaluating samples (workers=%d)...", workers)
    t0 = time.perf_counter()
    df, path = build_and_write(
        samples,
        args.out,
        metadata=meta,
        build_kwargs={
            "n_workers": workers,
            "chunksize": args.chunksize,
            "progress": not args.no_progress,
        },
    )
    elapsed = time.perf_counter() - t0

    n_ok = int((df["status"] == "ok").sum())
    n_total = len(df)
    n_failed = n_total - n_ok
    log.info(
        "wrote %d rows x %d cols to %s (ok=%d/%d, %.2f%%) in %.1f s (%.2f s/sample, "
        "%.2f s/sample/worker)",
        n_total,
        len(df.columns),
        path,
        n_ok,
        n_total,
        100 * n_ok / max(1, n_total),
        elapsed,
        elapsed / max(1, n_total),
        elapsed * workers / max(1, n_total),
    )
    if n_failed > 0:
        # Per-sample exceptions are recorded in the ``status`` column and
        # are part of the documented graceful-failure contract, not a
        # script-level error. Surface them so they're visible without
        # tripping CI.
        top_reasons = df.loc[df["status"] != "ok", "status"].value_counts().head(5)
        log.warning(
            "%d sample(s) hit a graceful-failure path (still recorded with NaN metrics). "
            "Top reasons: %s",
            n_failed,
            ", ".join(f"{k}={v}" for k, v in top_reasons.items()),
        )
    if "stalled" in df.columns:
        # Schema v6 (v6 schema update): infeasibility flag flipped from
        # ``motor_torque_ok`` to ``stalled`` (positive class = bad).
        # Report the *non-stalled* rate so the headline number stays
        # comparable to pre-v6 datasets ("higher is better").
        feas_rate = 1.0 - float(df["stalled"].astype(bool).mean())
        log.info("feasibility (non-stalled) rate: %.2f%%", 100 * feas_rate)

    # Non-zero exit only on catastrophic build failure (no rows written).
    # Per-sample graceful failures are by design and must not masquerade
    # as a script error, otherwise CI / shell wrappers misclassify
    # successful runs.
    return 0 if n_total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
