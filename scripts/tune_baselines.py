"""Optuna-tune the XGBoost surrogate baselines on a Parquet dataset.

Companion to ``scripts/run_baselines.py``. This script tunes only
XGBoost (per-target regressors + the ``stalled`` feasibility classifier);
the rationale for the scope is in
``roverdevkit.surrogate.tuning`` module docstring.

Outputs (under ``--out-dir``):

- ``tuned_summary.csv`` — one row per ``(target, kind)`` with the val
  objective the tuner achieved, the test-set metric on the refit
  model, and the tuning wall-clock.
- ``tuned_best_params.json`` — best hyperparameters per target,
  including the early-stopping-best ``n_estimators``.
- ``tuned_test_metrics.parquet`` — long-format ``(target, metric, value,
  scenario_family)`` frame for the tuned models on the test split,
  schema-compatible with the untuned ``metrics_long.parquet`` so a
  sibling Notebook / table can concat them.
- ``study_<target>.csv`` — ``study.trials_dataframe()`` per target
  for the writeup (objective trace, parameter samples, durations).
- ``tuned_registry_sanity.csv`` — Layer-1 registry-rover predictions
  for the tuned models, same schema as ``run_baselines.py``'s
  ``registry_sanity.csv`` so primary vs diagnostic targets and
  ``is_primary`` are handled identically.

Examples
--------
::

    # Full v4 tuning run (50 trials per target, ~10-20 min on 8 cores)
    python scripts/tune_baselines.py \\
        --dataset data/analytical/lhs_v4.parquet \\
        --out-dir reports/week8_tuned_v4

    # Smoke (10 trials per target, no classifier, ~1 min)
    python scripts/tune_baselines.py \\
        --dataset data/analytical/lhs_v4.parquet \\
        --out-dir /tmp/tune_smoke \\
        --n-trials 10 --no-classifier
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from roverdevkit.surrogate.baselines import (
    ACCEPTANCE_GATES,
    LAYER1_PRIMARY_TARGETS,
    _row_for_registry_rover,  # type: ignore[reportPrivateUsage]
)
from roverdevkit.surrogate.dataset import read_parquet
from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    PRIMARY_REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
    build_feature_matrix,
    valid_rows,
)
from roverdevkit.surrogate.tuning import (
    TuningResult,
    tune_xgboost_classifier,
    tune_xgboost_regressor,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna trials per target (default 50).",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Per-target tuning wall-clock cap. Default: no cap.",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=PRIMARY_REGRESSION_TARGETS,
        help="Regression targets to tune. Defaults to the four primary targets.",
    )
    p.add_argument(
        "--no-classifier",
        action="store_true",
        help="Skip tuning the stalled feasibility classifier.",
    )
    p.add_argument(
        "--no-registry-check",
        action="store_true",
        help="Skip the tuned registry-rover Layer-1 sanity check.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Plumbed through to XGBoost (-1 = all cores).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _split_xy(
    df: pd.DataFrame, target: str, *, feasible_only: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the (X, y) view for one target on a single split.

    Regression targets see only feasible rows; classification sees all
    valid (status == 'ok') rows.
    """
    df_clean = valid_rows(df)
    if feasible_only:
        # Schema v6 (W12 step B): ``FEASIBILITY_COLUMN`` is now ``stalled``
        # with positive class = infeasible, so we negate before masking
        # to keep only the feasible (non-stalled) regression rows. The
        # classifier path keeps the raw 0/1 labels (1 = stalled = the
        # positive failure class).
        mask = (~df_clean[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
        df_clean = df_clean.loc[mask]
    X = build_feature_matrix(df_clean)
    y = df_clean[target].to_numpy()
    if not feasible_only:
        y = y.astype(int)
    return X, y


def _regression_metrics_with_family(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    target: str,
    algorithm: str,
) -> pd.DataFrame:
    """Mirror the per-family metric layout in ``evaluate_baselines``."""
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("__all__", df_test)]
    if "scenario_family" in df_test.columns:
        for fam, sub in df_test.groupby("scenario_family", observed=True):
            groups.append((str(fam), sub))
    for fam, sub in groups:
        idx = df_test.index.isin(sub.index)
        y_true_g = df_test.loc[idx, target].to_numpy()
        y_pred_g = y_pred[idx]
        if len(y_true_g) < 2:
            continue
        metrics = {
            "r2": float(r2_score(y_true_g, y_pred_g)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_g, y_pred_g))),
            "mape": float(mean_absolute_percentage_error(y_true_g, y_pred_g)),
            "n": float(len(y_true_g)),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "target": target,
                    "split": "test",
                    "scenario_family": fam,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics_with_family(
    df_test: pd.DataFrame,
    y_score: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_true = df_test[FEASIBILITY_COLUMN].astype(int).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    groups: list[tuple[str, pd.DataFrame]] = [("__all__", df_test)]
    if "scenario_family" in df_test.columns:
        for fam, sub in df_test.groupby("scenario_family", observed=True):
            groups.append((str(fam), sub))
    for fam, sub in groups:
        idx = df_test.index.isin(sub.index)
        y_true_g = y_true[idx]
        y_score_g = y_score[idx]
        y_pred_g = y_pred[idx]
        if len(y_true_g) < 2:
            continue
        auc = (
            float("nan")
            if len(np.unique(y_true_g)) < 2
            else float(roc_auc_score(y_true_g, y_score_g))
        )
        metrics = {
            "auc": auc,
            "f1": float(f1_score(y_true_g, y_pred_g, zero_division=0)),
            "accuracy": float((y_pred_g == y_true_g).mean()),
            "n": float(len(y_true_g)),
            "positive_rate": float(y_true_g.mean()),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "algorithm": "xgboost_tuned",
                    "target": FEASIBILITY_COLUMN,
                    "split": "test",
                    "scenario_family": fam,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _build_training_categories(df: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Mirror ``fit_baselines``' captured-categories logic so the tuned
    registry-rover sanity check uses the same codebook as the untuned
    one."""
    out: dict[str, tuple[str, ...]] = {}
    df_clean = valid_rows(df)
    X_all = build_feature_matrix(df_clean)
    for col in SCENARIO_CATEGORICAL_COLUMNS:
        if col in X_all.columns:
            uniq = X_all[col].astype(str).unique()
            out[col] = tuple(sorted(str(x) for x in uniq))
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("tune_baselines")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("loading dataset from %s", args.dataset)
    df = read_parquet(args.dataset)
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]
    log.info("train=%d val=%d test=%d", len(df_train), len(df_val), len(df_test))

    summary_rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    metrics_frames: list[pd.DataFrame] = []
    fitted_regressors: dict[str, Any] = {}
    fitted_classifier: Any | None = None

    # --- regression tuning loop ----------------------------------------
    for target in args.targets:
        log.info("[regressor] tuning target=%s (n_trials=%d)", target, args.n_trials)
        X_tr, y_tr = _split_xy(df_train, target, feasible_only=True)
        X_va, y_va = _split_xy(df_val, target, feasible_only=True)
        X_te, y_te = _split_xy(df_test, target, feasible_only=True)

        result: TuningResult = tune_xgboost_regressor(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target=target,
            n_trials=args.n_trials,
            timeout_seconds=args.timeout_seconds,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        # Score on test
        y_te_pred = np.asarray(result.final_model.predict(X_te))
        df_test_feas = valid_rows(df_test)
        # Schema v6: negate ``stalled`` (True == infeasible) to keep the
        # feasible-only test rows the regression metrics expect.
        feas_mask = (~df_test_feas[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
        df_test_feas = df_test_feas.loc[feas_mask]
        m = _regression_metrics_with_family(
            df_test_feas, y_te_pred, target=target, algorithm="xgboost_tuned"
        )
        metrics_frames.append(m)
        fitted_regressors[target] = result.final_model

        test_overall = m.query("scenario_family == '__all__'").set_index("metric")["value"]
        log.info(
            "  done in %.1fs over %d trials; val R²=%.4f, test R²=%.4f, RMSE=%.3f",
            result.elapsed_seconds,
            result.n_trials,
            result.val_score,
            float(test_overall.get("r2", float("nan"))),
            float(test_overall.get("rmse", float("nan"))),
        )

        summary_rows.append(
            {
                "target": target,
                "kind": "regressor",
                "n_trials": result.n_trials,
                "tuning_seconds": result.elapsed_seconds,
                "val_objective": result.val_score,
                "val_objective_metric": "r2",
                "test_r2": float(test_overall.get("r2", float("nan"))),
                "test_rmse": float(test_overall.get("rmse", float("nan"))),
                "test_mape": float(test_overall.get("mape", float("nan"))),
                "best_n_estimators": int(result.best_params.get("n_estimators", -1)),
                "best_max_depth": int(result.best_params.get("max_depth", -1)),
                "best_learning_rate": float(result.best_params.get("learning_rate", float("nan"))),
            }
        )
        best_params[target] = {k: _coerce_for_json(v) for k, v in result.best_params.items()}
        # Persist the trial frame
        result.study_df.to_csv(args.out_dir / f"study_{target}.csv", index=False)

    # --- classifier tuning ---------------------------------------------
    if not args.no_classifier:
        log.info("[classifier] tuning target=%s", FEASIBILITY_COLUMN)
        X_tr, y_tr = _split_xy(df_train, FEASIBILITY_COLUMN, feasible_only=False)
        X_va, y_va = _split_xy(df_val, FEASIBILITY_COLUMN, feasible_only=False)
        X_te, y_te = _split_xy(df_test, FEASIBILITY_COLUMN, feasible_only=False)

        result_cls: TuningResult = tune_xgboost_classifier(
            X_tr,
            y_tr,
            X_va,
            y_va,
            target=FEASIBILITY_COLUMN,
            n_trials=args.n_trials,
            timeout_seconds=args.timeout_seconds,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        fitted_classifier = result_cls.final_model
        y_te_score = np.asarray(result_cls.final_model.predict_proba(X_te))[:, 1]
        df_test_clean = valid_rows(df_test)
        m = _classification_metrics_with_family(df_test_clean, y_te_score)
        metrics_frames.append(m)

        test_overall = m.query("scenario_family == '__all__'").set_index("metric")["value"]
        log.info(
            "  done in %.1fs over %d trials; val AUC=%.4f, test AUC=%.4f, F1=%.4f",
            result_cls.elapsed_seconds,
            result_cls.n_trials,
            result_cls.val_score,
            float(test_overall.get("auc", float("nan"))),
            float(test_overall.get("f1", float("nan"))),
        )

        summary_rows.append(
            {
                "target": FEASIBILITY_COLUMN,
                "kind": "classifier",
                "n_trials": result_cls.n_trials,
                "tuning_seconds": result_cls.elapsed_seconds,
                "val_objective": result_cls.val_score,
                "val_objective_metric": "auc",
                "test_auc": float(test_overall.get("auc", float("nan"))),
                "test_f1": float(test_overall.get("f1", float("nan"))),
                "test_accuracy": float(test_overall.get("accuracy", float("nan"))),
                "best_n_estimators": int(result_cls.best_params.get("n_estimators", -1)),
                "best_max_depth": int(result_cls.best_params.get("max_depth", -1)),
                "best_learning_rate": float(
                    result_cls.best_params.get("learning_rate", float("nan"))
                ),
            }
        )
        best_params[FEASIBILITY_COLUMN] = {
            k: _coerce_for_json(v) for k, v in result_cls.best_params.items()
        }
        result_cls.study_df.to_csv(args.out_dir / f"study_{FEASIBILITY_COLUMN}.csv", index=False)

    # --- write reports -------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "tuned_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path)

    params_path = args.out_dir / "tuned_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))
    log.info("wrote %s", params_path)

    if metrics_frames:
        metrics_long = pd.concat(metrics_frames, ignore_index=True)
        metrics_path = args.out_dir / "tuned_test_metrics.parquet"
        metrics_long.to_parquet(metrics_path, index=False)
        log.info("wrote %s (%d rows)", metrics_path, len(metrics_long))

        # Acceptance summary against the project plan thresholds
        gate_rows: list[dict[str, Any]] = []
        for tgt, thresholds in ACCEPTANCE_GATES.items():
            sub = metrics_long.query("target == @tgt and scenario_family == '__all__'").set_index(
                "metric"
            )["value"]
            row = {"target": tgt, "thresholds": json.dumps(thresholds)}
            passes = True
            for m_name, threshold in thresholds.items():
                v = float(sub.get(m_name, float("nan")))
                row[f"{m_name}_observed"] = v
                row[f"{m_name}_threshold"] = threshold
                passes = passes and not np.isnan(v) and v >= threshold
            row["passes"] = passes
            gate_rows.append(row)
        gate_df = pd.DataFrame(gate_rows)
        gate_path = args.out_dir / "tuned_acceptance_gate.csv"
        gate_df.to_csv(gate_path, index=False)
        log.info(
            "wrote %s; tuned passes %d/%d", gate_path, int(gate_df["passes"].sum()), len(gate_df)
        )
        print("\n=== Tuned XGBoost acceptance gate (test, all families) ===", flush=True)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(gate_df.to_string(index=False))

    print("\n=== Tuned XGBoost summary ===", flush=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))

    # --- Layer-1 registry-rover sanity check ---------------------------
    if not args.no_registry_check and (fitted_regressors or fitted_classifier is not None):
        log.info("running tuned registry-rover sanity check...")
        try:
            sanity = _tuned_registry_sanity(df, fitted_regressors, fitted_classifier)
            sanity_path = args.out_dir / "tuned_registry_sanity.csv"
            sanity.to_csv(sanity_path, index=False)
            log.info("wrote %s (%d rows)", sanity_path, len(sanity))
            _print_registry_summary(sanity)
        except Exception as exc:  # pragma: no cover — diagnostic, not fatal
            log.warning("tuned registry-rover sanity check failed: %s", exc)

    return 0


def _coerce_for_json(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _tuned_registry_sanity(
    df: pd.DataFrame,
    regressors: dict[str, Any],
    classifier: Any | None,
) -> pd.DataFrame:
    """Apply tuned models to the registry-rover Layer-1 inputs."""
    training_categories = _build_training_categories(df)
    primary_targets = set(LAYER1_PRIMARY_TARGETS)
    rovers = ("Pragyan", "Yutu-2", "MoonRanger", "Rashid-1")
    rows: list[dict[str, Any]] = []
    for rover in rovers:
        X_row, evaluator_metrics = _row_for_registry_rover(
            rover, training_categories=training_categories
        )
        for target, model in regressors.items():
            y_hat = float(np.asarray(model.predict(X_row))[0])
            y_true = float(evaluator_metrics[target])
            rows.append(
                {
                    "rover": rover,
                    "algorithm": "xgboost_tuned",
                    "target": target,
                    "predicted": y_hat,
                    "evaluator": y_true,
                    "abs_error": y_hat - y_true,
                    "rel_error": (y_hat - y_true) / y_true if y_true != 0 else float("nan"),
                    "is_primary": target in primary_targets,
                }
            )
        if classifier is not None:
            p = float(np.asarray(classifier.predict_proba(X_row))[0, 1])
            y_true_bool = bool(evaluator_metrics[FEASIBILITY_COLUMN])
            rows.append(
                {
                    "rover": rover,
                    "algorithm": "xgboost_tuned",
                    "target": FEASIBILITY_COLUMN,
                    "predicted": p,
                    "evaluator": float(y_true_bool),
                    "abs_error": p - float(y_true_bool),
                    "rel_error": float("nan"),
                    "is_primary": FEASIBILITY_COLUMN in primary_targets,
                }
            )
    return pd.DataFrame(rows)


def _print_registry_summary(sanity: pd.DataFrame) -> None:
    primary = sanity[sanity["is_primary"]]
    diagnostic = sanity[~sanity["is_primary"]]
    print("\n=== Tuned registry sanity (PRIMARY) ===", flush=True)
    reg = primary[primary["target"] != FEASIBILITY_COLUMN]
    if not reg.empty:
        s = (
            reg.assign(abs_pct=lambda d: 100 * d["rel_error"].abs())
            .groupby(["rover", "target"])["abs_pct"]
            .median()
            .unstack("target")
        )
        print("Median |rel_error| (%):")
        print(s.round(2).to_string())
    clf = primary[primary["target"] == FEASIBILITY_COLUMN]
    if not clf.empty:
        s = (
            clf.assign(
                hit=lambda d: (d["predicted"] >= 0.5).astype(int) == d["evaluator"].astype(int)
            )
            .groupby("rover")["hit"]
            .mean()
            .rename("classifier_accuracy")
            .to_frame()
        )
        print("\nClassifier accuracy (stalled):")
        print(s.round(3).to_string())
    if not diagnostic.empty:
        print("\n=== Tuned registry sanity (SCENARIO-OOD diagnostic) ===", flush=True)
        s = (
            diagnostic.assign(abs_pct=lambda d: 100 * d["rel_error"].abs())
            .groupby(["rover", "target"])["abs_pct"]
            .median()
            .unstack("target")
        )
        print("Median |rel_error| (%):")
        print(s.round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
