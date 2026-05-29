"""SCM-correction gate sweep: BW vs BW+correction on a small LHS sample.

Generates the formal SCM-correction gate gate input. For each LHS draw produced by
:func:`roverdevkit.surrogate.sampling.generate_samples` we evaluate the
mission **twice** — once on the analytical (Bekker-Wong only) path and
once with the wheel-level SCM correction composed in via
:mod:`roverdevkit.mission.traverse_sim` — and write a paired-metrics
parquet plus a tabular gate summary.

Gate criterion
----------------------------------------
The wheel-level correction is *not* automatically promoted into the
analytical surrogate. It is promoted only if the mission-level
discrepancy clears one of two thresholds:

- **Range:** median ``|range_scm − range_bw| / range_bw`` across feasible
  designs > 10 % within at least one scenario family. Below 10 %, BW is
  good enough at the mission level and SCM is reported as a §9 bounded
  sensitivity instead.
- **Sign bias:** > 5 % of paired runs disagree on the binary feasibility
  flag (``stalled``) or on the sign of ``energy_margin_raw_pct``.
  Sign flips mean the BW surrogate would mislabel the feasibility frontier
  even on rank-only criteria.

The script prints a verdict per criterion and an overall recommendation.
The decision itself is recorded in `reports/scm_correction_gate/` after the
sweep — this script just gathers the evidence.

Usage
-----
Smoke run (8 designs total, serial, fast)::

    python scripts/run_gate_sweep.py --n-per-scenario 2 --workers 1 \\
        --out data/scm/gate_eval_smoke.parquet

Production gate sweep (200 designs, 4 workers)::

    python scripts/run_gate_sweep.py --n-per-scenario 50 --workers 4 \\
        --out data/scm/gate_eval_v1.parquet
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roverdevkit.mission.evaluator import evaluate_verbose  # noqa: E402
from roverdevkit.surrogate.sampling import LHSSample, generate_samples  # noqa: E402
from roverdevkit.terramechanics.correction_model import (  # noqa: E402
    DEFAULT_CORRECTION_PATH,
    WheelLevelCorrection,
)

# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

# Per-process cache so each worker loads the joblib once on first call,
# not once per sample. The pool's "spawn" context creates a fresh module
# per process, so this global is genuinely per-process.
_CORRECTION: WheelLevelCorrection | None = None
_CORRECTION_PATH: Path | None = None


def _get_correction(path: Path) -> WheelLevelCorrection:
    global _CORRECTION, _CORRECTION_PATH
    if _CORRECTION is None or path != _CORRECTION_PATH:
        _CORRECTION = WheelLevelCorrection.load(path)
        _CORRECTION_PATH = path
    return _CORRECTION


_HEADLINE_FIELDS: tuple[str, ...] = (
    "range_km",
    "energy_margin_pct",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "peak_motor_torque_nm",
    "sinkage_max_m",
    "thermal_survival",
    "stalled",
)


def _evaluate_pair(args: tuple[LHSSample, str]) -> dict[str, float | int | str | bool]:
    """Evaluate one LHS sample under BW-only and BW+correction.

    Returns a flat dict with one ``bw_*`` and one ``scm_*`` column per
    headline metric, plus the LHS metadata needed to join back to the
    design table later.
    """
    sample, correction_path = args
    correction = _get_correction(Path(correction_path))

    bw = evaluate_verbose(
        sample.design,
        sample.scenario,
        soil_override=sample.soil,
    )
    scm = evaluate_verbose(
        sample.design,
        sample.scenario,
        soil_override=sample.soil,
        correction=correction,
    )

    row: dict[str, float | int | str | bool] = {
        "sample_index": sample.sample_index,
        "scenario_family": sample.scenario_family,
        "split": sample.split,
        "stratum_id": sample.stratum_id,
    }
    for f in _HEADLINE_FIELDS:
        row[f"bw_{f}"] = getattr(bw.metrics, f)
        row[f"scm_{f}"] = getattr(scm.metrics, f)
    return row


# ----------------------------------------------------------------------
# Gate summary
# ----------------------------------------------------------------------


def _safe_rel_err(scm: pd.Series, bw: pd.Series) -> pd.Series:
    """``|scm - bw| / |bw|`` with a small floor so range_km ≈ 0 doesn't blow up."""
    denom = bw.abs().clip(lower=1e-3)
    return (scm - bw).abs() / denom


_MOBILE_RANGE_KM_FLOOR: float = 0.1
"""Designs with range < 100 m under BOTH evaluators are treated as
'stalled / infeasible' for the purposes of relative-range comparison.
A BW-stalled / SCM-mobile pair is *not* a 'range disagreement' — it's a
*feasibility* disagreement, captured by ``feasibility_flip_frac``."""


def gate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario-family gate metrics.

    Two qualitatively different kinds of BW-vs-SCM disagreement are
    reported separately because they trigger different responses:

    - **Quantitative (range):** among designs feasible-in-both
      (range > 100 m under both evaluators), what is the median /
      p90 relative range delta? > 10 % median ⇒ ``range_gate=True``.
    - **Qualitative (feasibility):** how often do BW and SCM disagree
      on whether a design is feasible at all (``stalled == False`` AND
      ``range > 100 m``), and how often do they flip the sign of the
      mission energy balance? Either > 5 % ⇒ ``sign_gate=True``.

    Range Spearman ρ is reported on the both-mobile subset since ranks
    among "all stalled" designs are meaningless.
    """
    rows: list[dict[str, float | int | str | bool]] = []
    for family, sub in df.groupby("scenario_family", sort=True):
        both_mobile = (sub["bw_range_km"] > _MOBILE_RANGE_KM_FLOOR) & (
            sub["scm_range_km"] > _MOBILE_RANGE_KM_FLOOR
        )
        n_mobile = int(both_mobile.sum())
        if n_mobile > 0:
            rel_err = _safe_rel_err(
                sub.loc[both_mobile, "scm_range_km"], sub.loc[both_mobile, "bw_range_km"]
            )
            med = float(rel_err.median())
            p90 = float(rel_err.quantile(0.90))
        else:
            med = float("nan")
            p90 = float("nan")

        # Schema v6: ``stalled`` is the *infeasibility* flag (positive = bad);
        # negate to get the v5 ``motor_torque_ok`` semantics this gate
        # condition was originally written against.
        bw_feas = (~sub["bw_stalled"]) & (sub["bw_range_km"] > _MOBILE_RANGE_KM_FLOOR)
        scm_feas = (~sub["scm_stalled"]) & (sub["scm_range_km"] > _MOBILE_RANGE_KM_FLOOR)
        feas_flip = float((bw_feas != scm_feas).mean())

        em_sign_flip = float(
            (
                np.sign(sub["bw_energy_margin_raw_pct"])
                != np.sign(sub["scm_energy_margin_raw_pct"])
            ).mean()
        )

        spearman = float("nan")
        if n_mobile >= 5:
            sub_mob = sub[both_mobile]
            spearman = float(spearmanr(sub_mob["bw_range_km"], sub_mob["scm_range_km"]).statistic)

        rows.append(
            {
                "scenario_family": str(family),
                "n": int(len(sub)),
                "n_both_mobile": n_mobile,
                "range_med_rel_err": med,
                "range_p90_rel_err": p90,
                "feasibility_flip_frac": feas_flip,
                "sign_flip_energy_margin_frac": em_sign_flip,
                "range_spearman": spearman,
                "range_gate": (n_mobile >= 5) and (med > 0.10),
                "sign_gate": (feas_flip > 0.05) or (em_sign_flip > 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("scenario_family")


def overall_verdict(summary: pd.DataFrame) -> tuple[bool, list[str]]:
    """Apply the §6 SCM-correction gate rule: gate fires if any family triggers either threshold.

    Returns ``(gate_fires, reasons)``.
    """
    reasons: list[str] = []
    for family, row in summary.iterrows():
        if row["range_gate"]:
            reasons.append(
                f"{family}: median |Δrange|/range = {row['range_med_rel_err']:.1%} > 10%"
            )
        if row["sign_gate"]:
            reasons.append(
                f"{family}: feas-flip={row['feasibility_flip_frac']:.1%}, "
                f"em-sign-flip={row['sign_flip_energy_margin_frac']:.1%} (>5% threshold)"
            )
    return (len(reasons) > 0, reasons)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-per-scenario", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--correction",
        type=Path,
        default=DEFAULT_CORRECTION_PATH,
        help="Path to a WheelLevelCorrection joblib (default: production artifact)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/scm/gate_eval_v1.parquet"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    correction_path: Path = args.correction
    if not correction_path.exists():
        print(
            f"ERROR: correction artifact not found at {correction_path}. "
            "Run scripts/train_correction_model.py first.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading correction artifact:    {correction_path}")
    print(
        f"Generating LHS samples:          n_per_scenario={args.n_per_scenario}, seed={args.seed}"
    )
    samples = generate_samples(args.n_per_scenario, seed=args.seed)
    print(
        f"  → {len(samples)} samples across {len(set(s.scenario_family for s in samples))} families"
    )

    payload = [(s, correction_path) for s in samples]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating BW + SCM pairs:       workers={args.workers}")
    t0 = time.time()
    rows: list[dict[str, float | int | str | bool]] = []
    if args.workers <= 1:
        for i, item in enumerate(payload, 1):
            rows.append(_evaluate_pair(item))
            if i % 10 == 0:
                print(f"  {i}/{len(payload)}  ({time.time() - t0:.1f}s elapsed)")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = [pool.submit(_evaluate_pair, item) for item in payload]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 10 == 0:
                    print(f"  {i}/{len(payload)}  ({time.time() - t0:.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({elapsed / max(1, len(payload)):.2f}s per (BW, SCM) pair)")

    df = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote paired metrics:            {args.out}  ({len(df)} rows)")

    # Gate summary
    summary = gate_summary(df)
    print("\nPer-family gate summary:")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 160)
    print(summary.to_string())

    fires, reasons = overall_verdict(summary)
    print("\nOverall verdict:")
    if fires:
        print("  GATE FIRES — fit a mission-level correction surrogate. Triggers:")
        for r in reasons:
            print(f"    - {r}")
    else:
        print("  GATE DOES NOT FIRE — analytical-only surrogate is sufficient at the")
        print("  mission level. Report SCM as a §9 bounded sensitivity (median |Δrange|/range")
        print("  < 10% and feasibility-flip < 5% across all scenario families).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
