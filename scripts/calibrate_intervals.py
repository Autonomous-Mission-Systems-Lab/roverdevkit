"""Fit quantile XGBoost heads and calibrate 90 % prediction intervals.

Companion to ``scripts/tune_baselines.py`` for project_plan.md §6 / W8
step-4. Reads the W8 step-3 tuned hyperparameters from
``--tuned-params`` (default
``reports/week8_tuned_v4/tuned_best_params.json``), refits each
primary regression target as three quantile heads
(``τ ∈ {0.05, 0.50, 0.95}``) on the v4 corpus, and reports empirical
90 % coverage and PI width on the canonical test split overall and per
scenario family.

Outputs (under ``--out-dir``):

- ``coverage.csv`` — long-format coverage / width / crossing-rate
  frame. One row per ``(target, scenario_family, repair)``.
- ``median_sanity.csv`` — τ=0.5 head test R² vs the W8 step-3 tuned
  median R² as the §6.2 sanity guardrail.
- ``quantile_bundles.joblib`` — dict ``{target: QuantileHeads}``
  serialised together for downstream NSGA-II / Pareto-uncertainty
  consumers.
- ``fit_seconds.csv`` — per-target wall-clock for the three-head fit.

Examples
--------
::

    # Full v4 calibration (≈3-6 min on 8 cores)
    python scripts/calibrate_intervals.py \\
        --dataset data/analytical/lhs_v4.parquet \\
        --tuned-params reports/week8_tuned_v4/tuned_best_params.json \\
        --out-dir reports/week8_intervals_v4

    # Smoke (single target, no save)
    python scripts/calibrate_intervals.py \\
        --dataset data/analytical/lhs_v4.parquet \\
        --tuned-params reports/week8_tuned_v4/tuned_best_params.json \\
        --out-dir /tmp/intervals_smoke \\
        --targets range_km
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from roverdevkit.surrogate.dataset import read_parquet
from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    PRIMARY_REGRESSION_TARGETS,
    build_feature_matrix,
    valid_rows,
)
from roverdevkit.surrogate.uncertainty import (
    DEFAULT_QUANTILES,
    QuantileHeads,
    coverage_table,
    fit_quantile_heads,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--tuned-params",
        type=Path,
        required=True,
        help="Path to tuned_best_params.json from W8 step-3.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--targets",
        nargs="+",
        default=PRIMARY_REGRESSION_TARGETS,
        help="Primary regression targets to calibrate. Default: all four.",
    )
    p.add_argument(
        "--quantiles",
        nargs=3,
        type=float,
        default=list(DEFAULT_QUANTILES),
        metavar=("LOW", "MID", "HI"),
        help="Quantile triple. Default: 0.05 0.50 0.95 (90% PI).",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=25,
        help="Patience on val pinball loss. Mirrors W8 step-3.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Build feasible-only (X, y, scenario_family) for one regression target."""
    df_clean = valid_rows(df)
    # Schema v6 (W12 step B): ``FEASIBILITY_COLUMN`` is now ``stalled``
    # with positive class = infeasible, so we negate before masking to
    # keep only the feasible (non-stalled) rows the regression heads
    # were trained on.
    mask = (~df_clean[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
    df_clean = df_clean.loc[mask]
    X = build_feature_matrix(df_clean)
    y = df_clean[target].to_numpy()
    fam = (
        df_clean["scenario_family"].astype(str).reset_index(drop=True)
        if "scenario_family" in df_clean.columns
        else pd.Series([], dtype=object)
    )
    return X.reset_index(drop=True), y, fam


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("calibrate_intervals")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading dataset from %s", args.dataset)
    df = read_parquet(args.dataset)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]
    log.info("train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    log.info("loading tuned hyperparameters from %s", args.tuned_params)
    tuned_params: dict[str, dict[str, Any]] = json.loads(args.tuned_params.read_text())

    bundles: dict[str, QuantileHeads] = {}
    coverage_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []

    quantiles = tuple(float(q) for q in args.quantiles)
    if not (quantiles[0] < quantiles[1] < quantiles[2]):
        raise SystemExit(f"--quantiles must be strictly increasing, got {quantiles}")

    for target in args.targets:
        if target not in tuned_params:
            log.warning(
                "no tuned params for %s; skipping (run scripts/tune_baselines.py first)",
                target,
            )
            continue
        log.info("[%s] fitting quantile heads at τ=%s", target, quantiles)
        X_tr, y_tr, _ = _split_xy(df_train, target)
        X_va, y_va, _ = _split_xy(df_val, target)
        X_te, y_te, fam_te = _split_xy(df_test, target)

        bundle = fit_quantile_heads(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target=target,
            base_params=tuned_params[target],
            quantiles=quantiles,  # type: ignore[arg-type]
            early_stopping_rounds=args.early_stopping_rounds,
            n_jobs=args.n_jobs,
        )
        bundles[target] = bundle

        for repair in (False, True):
            cov = coverage_table(
                bundle,
                X_te,
                y_te,
                scenario_family=fam_te,
                repair_crossings=repair,
            )
            cov["repair"] = "sorted" if repair else "raw"
            coverage_frames.append(cov)

        # Sanity guardrail: median (τ=0.5) head R² vs W8 step-3 tuned R²
        preds = bundle.predict(X_te, repair_crossings=False)
        keys = list(preds.keys())  # q_lo, q_mid, q_hi
        y_pred_mid = preds[keys[1]]
        r2_mid = float(r2_score(y_te, y_pred_mid))

        cov_overall = (
            coverage_frames[-2]  # raw, overall
            .query("scenario_family == '__all__'")
            .iloc[0]
        )
        log.info(
            "[%s] τ=0.5 R²=%.4f (sanity); 90%% coverage=%.3f (raw), mean width=%.3f, "
            "crossings=%.2f%%; fit %.1fs",
            target,
            r2_mid,
            cov_overall["empirical"],
            cov_overall["mean_width"],
            100 * cov_overall["crossing_rate"],
            bundle.fit_seconds,
        )

        fit_rows.append(
            {
                "target": target,
                "fit_seconds": bundle.fit_seconds,
                "n_train": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
            }
        )
        sanity_rows.append(
            {
                "target": target,
                "median_test_r2": r2_mid,
                "step3_tuned_test_r2_path": str(args.tuned_params.parent / "tuned_summary.csv"),
            }
        )

    # ---- write reports ----------------------------------------------------
    coverage_path = args.out_dir / "coverage.csv"
    pd.concat(coverage_frames, ignore_index=True).to_csv(coverage_path, index=False)
    log.info("wrote %s", coverage_path)

    fit_path = args.out_dir / "fit_seconds.csv"
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    log.info("wrote %s", fit_path)

    # Append the W8 step-3 tuned R² for the same target if the report is on disk
    step3_summary_path = args.tuned_params.parent / "tuned_summary.csv"
    if step3_summary_path.exists():
        step3 = pd.read_csv(step3_summary_path)
        step3 = step3[step3["kind"] == "regressor"][["target", "test_r2"]].rename(
            columns={"test_r2": "step3_tuned_test_r2"}
        )
        sanity_df = pd.DataFrame(sanity_rows).merge(step3, on="target", how="left")
        sanity_df["delta_r2"] = sanity_df["median_test_r2"] - sanity_df["step3_tuned_test_r2"]
    else:
        sanity_df = pd.DataFrame(sanity_rows)
    sanity_path = args.out_dir / "median_sanity.csv"
    sanity_df.to_csv(sanity_path, index=False)
    log.info("wrote %s", sanity_path)

    bundles_path = args.out_dir / "quantile_bundles.joblib"
    joblib.dump(bundles, bundles_path)
    log.info("wrote %s (%d bundles)", bundles_path, len(bundles))

    # ---- console summary --------------------------------------------------
    cov_all = pd.concat(coverage_frames, ignore_index=True)
    cov_overall = cov_all.query("scenario_family == '__all__' and repair == 'raw'")
    print("\n=== 90% PI calibration summary (test split, raw quantile output) ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        cols = ["target", "n", "nominal", "empirical", "mean_width", "crossing_rate"]
        print(cov_overall[cols].round(4).to_string(index=False))

    print("\n=== Median (τ=0.5) sanity vs W8 step-3 tuned ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        keep = [
            c
            for c in ("target", "median_test_r2", "step3_tuned_test_r2", "delta_r2")
            if c in sanity_df.columns
        ]
        print(sanity_df[keep].round(4).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
