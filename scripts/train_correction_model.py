"""Fit the SCM-correction wheel-level SCM correction model on a gate-sweep parquet.

Reads ``data/scm/runs_v1.parquet`` (paired Bekker-Wong + SCM single-wheel
runs produced by ``scripts/run_scm_sweep.py``), fits Ridge / Random
Forest / XGBoost per delta target, picks the best-by-test-RMSE
algorithm per target, and writes both the joblib-saved
:class:`roverdevkit.terramechanics.correction_model.WheelLevelCorrection`
artifact and a tidy fit-summary CSV used by the SCM-correction gate gate report.

Usage
-----
::

    python scripts/train_correction_model.py \\
        --parquet data/scm/runs_v1.parquet \\
        --out data/scm/correction_v1.joblib \\
        --fit-summary reports/scm_correction_gate/correction_fit.csv \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from roverdevkit.terramechanics.correction_model import (
    ACCEPTANCE_R2,
    REGRESSION_ALGORITHMS,
    TARGET_COLUMNS,
    train_correction_model,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parquet", type=Path, required=True, help="gate-sweep parquet")
    p.add_argument("--out", type=Path, required=True, help="output joblib artifact")
    p.add_argument(
        "--fit-summary",
        type=Path,
        default=None,
        help="optional output CSV with per-(target, algorithm, split) metrics",
    )
    p.add_argument("--seed", type=int, default=42, help="random_state for splits and trees")
    p.add_argument("--n-jobs", type=int, default=1, help="trees-side parallelism")
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=list(REGRESSION_ALGORITHMS),
        choices=list(REGRESSION_ALGORITHMS),
        help="subset of algorithms to fit; default is all three",
    )
    return p.parse_args(argv)


def _print_summary(fit_summary: pd.DataFrame, model) -> None:
    """Tabular console summary keyed off the saved metadata."""
    chosen = model.metadata["chosen_per_target"]
    print()
    print("=" * 78)
    print("SCM-correction wheel-level correction model — fit summary")
    print("=" * 78)
    print(
        f"Train / val / test rows: "
        f"{model.metadata['n_train']} / {model.metadata['n_val']} / {model.metadata['n_test']}"
    )
    print()

    # Tidy: per-target test scores for every algorithm + the refit on train+val
    pivot = fit_summary[fit_summary["split"] == "test"].pivot_table(
        index="target", columns="algorithm", values=["r2", "rmse"]
    )
    print("Per-target test metrics (R² | RMSE):")
    for tgt in TARGET_COLUMNS:
        print(f"  {tgt}:  chosen → {chosen[tgt]}")
        if tgt in pivot.index:
            row = pivot.loc[tgt]
            for algo in REGRESSION_ALGORITHMS:
                if (("r2", algo) in row.index) and pd.notna(row[("r2", algo)]):
                    flag = "*" if algo == chosen[tgt] else " "
                    print(
                        f"    {flag} {algo:14s}  R²={row[('r2', algo)]:6.3f}  "
                        f"RMSE={row[('rmse', algo)]:8.4f}"
                    )
            refit_key = ("r2", f"{chosen[tgt]}_refit")
            if refit_key in row.index and pd.notna(row[refit_key]):
                print(
                    f"    + {chosen[tgt]}_refit (train+val)  "
                    f"R²={row[refit_key]:6.3f}  "
                    f"RMSE={row[('rmse', f'{chosen[tgt]}_refit')]:8.4f}"
                )
    print()

    # Acceptance gate (informative; not used to block the artifact)
    failing = []
    for tgt in TARGET_COLUMNS:
        refit_r2 = pivot.loc[tgt, ("r2", f"{chosen[tgt]}_refit")]
        if pd.notna(refit_r2) and refit_r2 < ACCEPTANCE_R2:
            failing.append((tgt, float(refit_r2)))
    if failing:
        print(
            f"NOTE: {len(failing)} target(s) below the informative R² floor of "
            f"{ACCEPTANCE_R2:.2f}: {failing}. The SCM-correction gate gate is decided on "
            f"mission-level rank correlation, not wheel-level R²."
        )
    else:
        print(f"All targets cleared the informative R² floor of {ACCEPTANCE_R2:.2f}.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"Loading gate-sweep parquet from {args.parquet}")
    model, fit_summary = train_correction_model(
        args.parquet,
        args.out,
        fit_summary_path=args.fit_summary,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        algorithms=tuple(args.algorithms),
    )
    print(f"Wrote correction artifact to {args.out}")
    if args.fit_summary is not None:
        print(f"Wrote fit summary to {args.fit_summary}")
    _print_summary(fit_summary, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
