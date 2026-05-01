"""Week-7.7 bake-off: BW vs BW+correction vs SCM-direct.

Evaluates the same LHS sample under three wheel-force backends and
treats SCM-direct as ground truth. Outputs paired metrics (parquet)
plus a summary CSV reporting MAE / relative error and feasibility-flip
rates of {BW, BW+correction} vs SCM-direct.

Why this is the gating experiment for the architecture decision:

* If BW+correction tracks SCM-direct within the surrogate's regression
  noise floor (~5% on key targets) and feasibility flips are within the
  per-scenario gate thresholds, the production architecture stays
  BW + wheel-level correction → fast 40 k-row dataset rebuild,
  cheap inference at tradespace-search time.
* If BW+correction has systematically larger residuals than the
  correction model's own test-set RMSE — i.e. mission-level integration
  amplifies wheel-level errors — the project switches to SCM-direct
  as the dataset-generation backend (see ``roverdevkit/mission/
  traverse_sim.py::_scm_solve_step_wheel_forces`` and the perf
  budget there).

Usage::

    uv run python scripts/run_bakeoff.py \\
        --n-samples 100 --workers 6 \\
        --out reports/week7_7_bakeoff/

Each sample evaluates ~30 ms (BW) + ~50 ms (BW+corr) + ~3 s
(SCM-direct), so 100 samples ≈ 6 min on 6 spawn workers.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from roverdevkit.mission.evaluator import evaluate_verbose
from roverdevkit.surrogate.sampling import generate_samples

_BACKENDS: tuple[str, ...] = ("bw", "bw_corr", "scm")


def _eval_one(args: tuple[int, str, dict]) -> dict:
    """Worker: evaluate one (sample, backend) pair, return flat row."""
    sample_id, backend, sample_payload = args
    # Late imports inside worker so spawn pickling stays cheap.
    from roverdevkit.schema import DesignVector, MissionScenario
    from roverdevkit.terramechanics.bekker_wong import SoilParameters

    design = DesignVector(**sample_payload["design"])
    scenario = MissionScenario(**sample_payload["scenario"])
    soil = SoilParameters(**sample_payload["soil"])

    if backend == "bw":
        kw = {"force_backend": "bw"}
    elif backend == "bw_corr":
        kw = {"force_backend": "bw", "use_scm_correction": True}
    elif backend == "scm":
        kw = {"force_backend": "scm"}
    else:
        raise ValueError(backend)

    t0 = time.time()
    base = {
        "sample_id": sample_id,
        "backend": backend,
        "scenario_family": sample_payload["scenario_family"],
        "status": "ok",
    }
    try:
        res = evaluate_verbose(design, scenario, soil_override=soil, **kw)
    except Exception as exc:  # noqa: BLE001 -- graceful-failure pattern (see roverdevkit/surrogate/dataset.py)
        base["status"] = type(exc).__name__
        base["eval_seconds"] = time.time() - t0
        for col in (
            "range_km",
            "energy_margin_raw_pct",
            "slope_capability_deg",
            "peak_motor_torque_nm",
            "sinkage_max_m",
        ):
            base[col] = float("nan")
        # Schema v6 (W12 step B): graceful-failure rows default to
        # ``stalled = True`` (the safe-conservative side, matching
        # surrogate.dataset._nan_outputs).
        base["stalled"] = True
        base["thermal_survival"] = False
        return base
    dt = time.time() - t0
    m = res.metrics
    base.update(
        {
            "range_km": m.range_km,
            "energy_margin_raw_pct": m.energy_margin_raw_pct,
            "slope_capability_deg": m.slope_capability_deg,
            "peak_motor_torque_nm": m.peak_motor_torque_nm,
            "sinkage_max_m": m.sinkage_max_m,
            "stalled": bool(m.stalled),
            "thermal_survival": bool(m.thermal_survival),
            "eval_seconds": dt,
        }
    )
    return base


def _safe_rel_err(pred: np.ndarray, truth: np.ndarray, floor: float) -> np.ndarray:
    """Relative error with a floor on the denominator. Zeros vs zeros = 0."""
    denom = np.maximum(np.abs(truth), floor)
    err = np.abs(pred - truth) / denom
    err = np.where((np.abs(truth) < floor) & (np.abs(pred) < floor), 0.0, err)
    return err


def _summarise(wide: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-scenario MAE / rel-err / flip rate vs SCM-direct."""
    rows = []
    targets_continuous = [
        ("range_km", 0.1),  # 100 m floor: avoids blow-ups when both ~0
        ("energy_margin_raw_pct", 1.0),
        ("peak_motor_torque_nm", 0.05),
        ("sinkage_max_m", 1e-3),
    ]
    flip_targets = ["stalled"]

    for fam, sub in wide.groupby("scenario_family"):
        n = len(sub)
        out = {"scenario_family": fam, "n": n}

        scm_mobile = sub["range_km_scm"] > 0.1
        for backend in ("bw", "bw_corr"):
            mobile = scm_mobile & (sub[f"range_km_{backend}"] > 0.1)
            for tgt, floor in targets_continuous:
                truth = sub[f"{tgt}_scm"].to_numpy()
                pred = sub[f"{tgt}_{backend}"].to_numpy()
                # Range / energy errors only meaningful where SCM mobile.
                mask = (
                    mobile.to_numpy()
                    if tgt in {"range_km", "energy_margin_raw_pct"}
                    else np.ones(n, dtype=bool)
                )
                if mask.sum() == 0:
                    out[f"{tgt}_{backend}_mae"] = np.nan
                    out[f"{tgt}_{backend}_relerr_p50"] = np.nan
                    out[f"{tgt}_{backend}_relerr_p90"] = np.nan
                    continue
                t = truth[mask]
                p = pred[mask]
                out[f"{tgt}_{backend}_mae"] = float(np.mean(np.abs(p - t)))
                rel = _safe_rel_err(p, t, floor)
                out[f"{tgt}_{backend}_relerr_p50"] = float(np.median(rel))
                out[f"{tgt}_{backend}_relerr_p90"] = float(np.quantile(rel, 0.9))
            for flip_tgt in flip_targets:
                disagreement = (sub[f"{flip_tgt}_{backend}"] != sub[f"{flip_tgt}_scm"]).mean()
                out[f"{flip_tgt}_{backend}_flip_frac"] = float(disagreement)

        rows.append(out)
    return pd.DataFrame(rows).sort_values("scenario_family").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="LHS designs to draw via generate_samples (each design "
        "× scenarios produces multiple rows).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("reports/week7_7_bakeoff"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    samples = generate_samples(args.n_samples, seed=args.seed)
    print(
        f"[bakeoff] {len(samples)} (design, scenario) pairs across families: "
        f"{sorted({s.scenario_family for s in samples})}"
    )

    payloads = []
    for i, s in enumerate(samples):
        payload = {
            "design": s.design.model_dump(),
            "scenario": s.scenario.model_dump(),
            "soil": asdict(s.soil),
            "scenario_family": s.scenario_family,
        }
        for backend in _BACKENDS:
            payloads.append((i, backend, payload))

    print(f"[bakeoff] dispatching {len(payloads)} eval tasks across {args.workers} spawn workers")
    ctx = get_context("spawn")
    rows = []
    t_start = time.time()
    last_report = t_start
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as exe:
        for i, row in enumerate(exe.map(_eval_one, payloads, chunksize=1), start=1):
            rows.append(row)
            now = time.time()
            if now - last_report > 15.0:
                pct = 100.0 * i / len(payloads)
                eta = (now - t_start) * (len(payloads) - i) / max(1, i)
                print(
                    f"  [{i}/{len(payloads)} {pct:5.1f}%] elapsed {now - t_start:.0f}s ETA {eta:.0f}s"
                )
                last_report = now
    print(f"[bakeoff] all evals complete in {time.time() - t_start:.0f}s")

    long_df = pd.DataFrame(rows)
    long_path = args.out / "bakeoff_long.parquet"
    long_df.to_parquet(long_path, index=False)
    n_failed = int((long_df["status"] != "ok").sum())
    if n_failed:
        top = long_df.loc[long_df["status"] != "ok", "status"].value_counts().head(5)
        print(
            f"[bakeoff] {n_failed}/{len(long_df)} evals hit graceful-failure path. Top reasons: "
            + ", ".join(f"{k}={v}" for k, v in top.items())
        )
    print(f"[bakeoff] wrote {long_path} ({len(long_df)} rows; {len(long_df) - n_failed} ok)")

    # Drop sample_ids where ANY backend failed; comparisons require all
    # three backends ok for the same sample.
    bad_sample_ids = set(long_df.loc[long_df["status"] != "ok", "sample_id"].unique())
    if bad_sample_ids:
        print(f"[bakeoff] dropping {len(bad_sample_ids)} samples with at least one failed backend")
        long_df = long_df.loc[~long_df["sample_id"].isin(bad_sample_ids)].copy()

    # Pivot to wide: one row per (sample_id, scenario_family),
    # columns suffixed by backend.
    metric_cols = [
        "range_km",
        "energy_margin_raw_pct",
        "slope_capability_deg",
        "peak_motor_torque_nm",
        "sinkage_max_m",
        "stalled",
        "thermal_survival",
        "eval_seconds",
    ]
    wide = long_df.pivot_table(
        index=["sample_id", "scenario_family"],
        columns="backend",
        values=metric_cols,
        aggfunc="first",
    )
    wide.columns = [f"{m}_{b}" for m, b in wide.columns]
    wide = wide.reset_index()
    wide_path = args.out / "bakeoff_wide.parquet"
    wide.to_parquet(wide_path, index=False)
    print(f"[bakeoff] wrote {wide_path}")

    summary = _summarise(wide)
    summary_path = args.out / "bakeoff_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[bakeoff] wrote {summary_path}")
    print()
    print("=" * 100)
    print("Per-scenario summary (vs SCM-direct as ground truth):")
    print("=" * 100)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:7.4f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
