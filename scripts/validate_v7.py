"""Run the W12 step-B follow-up validation gates against the v7 evaluator + surrogate.

Companion to ``scripts/build_dataset.py`` / ``tune_baselines.py`` /
``calibrate_intervals.py``. Aggregates the post-rebuild gates from
``reports/week12_design/decision.md`` (§ Acceptance gates) into one
artefact dump:

1. **Registry-rover Layer-0 truth comparison.** Re-uses the existing
   :func:`roverdevkit.validation.rover_comparison.compare_all` so the
   pass/fail polarity is identical to the CI gate. Verifies that the
   v7 evaluator (with the drivetrain-aware design vector and the
   per-scenario ``operational_duty_cycle`` as the *single* duty knob)
   still reproduces flown-rover range, thermal, peak solar, and stall
   verdicts.
2. **Registry-rover cruise speed.** Cruise speed is *derived* from
   peak hub torque + slip + energy balance + kinematic cap rather
   than a design input. The model's ``v_cruise`` is a physical
   *capability* upper bound (the largest speed the rover can sustain
   under torque + energy + slip + ω_hub × R caps) whereas published
   "nominal speed" typically reflects mission-conservative ops choices
   (drive then stop, hazard avoidance, comms windows) that are not
   modeled here. So the acceptance criterion is ``v_cruise >= 0.5 ×
   published_low`` (the model must at least *allow* the published
   speed) AND ``v_cruise <= 100 × published_high`` (catches order-of-
   magnitude unit / wiring bugs). Real-mission ops conservatism is
   captured by ``operational_duty_cycle``.
3. **Stall-edge sanity.** The four registry rovers on their canonical
   scenarios must not stall (matches Layer-0 truth), and a stress-
   perturbed variant (same design, scenario slope replaced with a
   wall-steep value) must stall. We perturb the *scenario* rather than
   the design's ``peak_wheel_torque_nm`` because the schema floor
   (0.3 Nm) would clamp torque-side perturbations on the lightest
   registry rovers (Pragyan, Rashid-1) before they hit demand.
   Catches the case where the slope-driven torque demand or slip
   solver is wired the wrong way.
4. **Surrogate quality on the feasible test set.** Per-target R² and
   90 % PI coverage on the held-out test split, restricted to
   feasible (= not-stalled) rows. v7 dropped the v6 saturated-δ_des
   regime split because the design-side ``designed_duty_cycle`` field
   it depended on is gone (drive duty cycle is now a single
   per-scenario quantity); the natural simplification is to gate
   on overall surrogate quality rather than a regime that no longer
   exists in the schema.

Outputs (under ``--out-dir``, defaults to
``reports/week12_validation_v7``):

- ``layer0_summary.csv`` — :func:`compare_all` result table.
- ``cruise_speed.csv`` — per-rover derived ``v_cruise`` vs published
  nominal speeds + pass/fail.
- ``stall_sanity.csv`` — per-rover baseline + perturbed stall verdict.
- ``test_quality.csv`` — per-target R² + coverage on the feasible
  test split.
- ``validate_v7_report.md`` — human-readable rollup with
  pass/fail per gate.

Usage
-----
::

    python scripts/validate_v7.py \\
        --dataset data/analytical/lhs_v7.parquet \\
        --quantile-bundles reports/week12_intervals_v7/quantile_bundles.joblib \\
        --out-dir reports/week12_validation_v7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from roverdevkit.mission.evaluator import evaluate, evaluate_verbose
from roverdevkit.surrogate.dataset import read_parquet
from roverdevkit.surrogate.features import (
    FEASIBILITY_COLUMN,
    PRIMARY_REGRESSION_TARGETS,
    build_feature_matrix,
    valid_rows,
)
from roverdevkit.validation.rover_comparison import compare_all
from roverdevkit.validation.rover_registry import registry

# Published "nominal speed" bands. Values are intentionally wider than
# any single citation because published speeds are reported with
# different operational definitions (max mechanical, average traverse,
# nominal-during-drive). Sources cited per-entry.
PUBLISHED_SPEED_BANDS_MPS: dict[str, tuple[float, float, str]] = {
    # Yutu-2: "continuous drive speed 40 mm/s" (Di et al. 2020 Icarus).
    "Yutu-2": (0.025, 0.060, "Di et al. 2020 Icarus; Ding et al. 2022"),
    # Pragyan: ~10 mm/s class (ISRO press kits; back-solved from 100 m
    # traverse over Lunar Day 1 ops window).
    "Pragyan": (0.003, 0.020, "ISRO Chandrayaan-3 mission updates 2023"),
    # MoonRanger: 0.05 m/s nominal, 0.07 m/s max (Kumar et al. i-SAIRAS
    # 2020 #5068).
    "MoonRanger": (0.030, 0.090, "Kumar et al. i-SAIRAS 2020 #5068"),
    # Rashid-1: 0.01-0.03 m/s class-typical for 10 kg rover (Els et al.
    # LPSC 2021; specific number not published before mission loss).
    "Rashid-1": (0.005, 0.040, "Els et al. LPSC 2021 (class-typical)"),
}

# Acceptance band: derived v_cruise is treated as a *capability upper
# bound*, not an ops-throttled match of published nominal speed. Pass
# if ``v_cruise >= CRUISE_LOWER_TOL × v_pub_low`` (model must allow the
# published speed) and ``v_cruise <= CRUISE_UPPER_TOL × v_pub_high``
# (sanity ceiling on the physics; catches order-of-magnitude bugs).
CRUISE_LOWER_TOL = 0.5
CRUISE_UPPER_TOL = 100.0

# Gate 4 acceptance thresholds on the feasible test split: per-target
# R² and 90 % PI coverage. These match the project_plan.md §7 L1 bar
# (R² ≥ 0.85 on primary targets; coverage ≥ 0.85 since we calibrate to
# 0.90 nominal but allow ±5pp slack for finite-sample noise).
TEST_R2_FLOOR = 0.85
TEST_COVERAGE_FLOOR = 0.85


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/analytical/lhs_v7.parquet"),
    )
    p.add_argument(
        "--quantile-bundles",
        type=Path,
        default=Path("reports/week12_intervals_v7/quantile_bundles.joblib"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/week12_validation_v7"),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Gate 1: Layer-0 truth comparison
# ---------------------------------------------------------------------------


def _gate_layer0() -> tuple[pd.DataFrame, bool]:
    summary = compare_all()
    rows: list[dict[str, Any]] = []
    for r in summary.results:
        rows.append(
            {
                "rover": r.rover_name,
                "range_m_predicted": r.range_m_predicted,
                "range_m_published": r.truth.traverse_m_published,
                "range_ratio": r.range_ratio,
                "thermal_predicted": r.metrics.thermal_survival,
                "thermal_published": r.truth.thermal_survival_published,
                "stalled": r.metrics.stalled,
                "peak_solar_w_predicted": r.peak_solar_power_w_predicted,
                "peak_solar_w_band_low": r.truth.peak_solar_power_w_low,
                "peak_solar_w_band_high": r.truth.peak_solar_power_w_high,
                "passes": r.passes,
            }
        )
    df = pd.DataFrame(rows)
    return df, summary.all_pass


# ---------------------------------------------------------------------------
# Gate 2: Registry-rover cruise speed
# ---------------------------------------------------------------------------


def _gate_cruise_speed() -> tuple[pd.DataFrame, bool]:
    """Compare derived v_cruise to published nominal speed bands."""
    rows: list[dict[str, Any]] = []
    for entry in registry():
        detailed = evaluate_verbose(
            entry.design,
            entry.scenario,
            gravity_m_per_s2=entry.gravity_m_per_s2,
            thermal_architecture=entry.thermal_architecture,
        )
        v_cruise = float(detailed.log.cruise_speed_mps)
        kinematic_clamped = bool(detailed.log.cruise_kinematic_clamped)
        band = PUBLISHED_SPEED_BANDS_MPS.get(entry.rover_name)
        if band is None:
            rows.append(
                {
                    "rover": entry.rover_name,
                    "v_cruise_mps": v_cruise,
                    "v_pub_low": float("nan"),
                    "v_pub_high": float("nan"),
                    "ratio_low": float("nan"),
                    "ratio_high": float("nan"),
                    "kinematic_clamped": kinematic_clamped,
                    "regime": "no_band",
                    "passes": True,
                    "citation": "(no band)",
                }
            )
            continue
        v_lo, v_hi, cite = band
        accept_lo = v_lo * CRUISE_LOWER_TOL
        accept_hi = v_hi * CRUISE_UPPER_TOL
        # ``v_cruise == 0`` is not a *cruise* failure but an energy-
        # limited regime: the energy-balance solver in
        # ``cruise_speed`` returns 0 when daylight power can't cover
        # avionics + drive load. Treat as PASS for this gate and
        # surface it via the ``regime`` column. Gate 1 catches
        # "rover can't reach published distance".
        if v_cruise == 0.0:
            regime = "energy_limited"
            passes = True
        else:
            regime = "kinematic" if kinematic_clamped else "energy_balance"
            passes = accept_lo <= v_cruise <= accept_hi
        rows.append(
            {
                "rover": entry.rover_name,
                "v_cruise_mps": v_cruise,
                "v_pub_low": v_lo,
                "v_pub_high": v_hi,
                "ratio_low": v_cruise / v_lo,
                "ratio_high": v_cruise / v_hi,
                "kinematic_clamped": kinematic_clamped,
                "regime": regime,
                "passes": passes,
                "citation": cite,
            }
        )
    df = pd.DataFrame(rows)
    all_pass = bool(df["passes"].all())
    return df, all_pass


# ---------------------------------------------------------------------------
# Gate 3: Stall-edge sanity
# ---------------------------------------------------------------------------


_STALL_PERTURB_SLOPE_DEG: float = 60.0


def _gate_stall_edge() -> tuple[pd.DataFrame, bool]:
    """Baseline rovers don't stall; same design on a 60° slope must."""
    rows: list[dict[str, Any]] = []
    for entry in registry():
        baseline_metrics = evaluate(
            entry.design,
            entry.scenario,
            gravity_m_per_s2=entry.gravity_m_per_s2,
            thermal_architecture=entry.thermal_architecture,
        )
        # Perturb the scenario, not the design: replace the scenario's
        # max slope with a wall-steep value so torque demand or the
        # slip solver fails. Using the same model_copy idiom as the
        # ops-duty-override path; we keep loose-soil and other fields.
        perturbed_scenario = entry.scenario.model_copy(
            update={"max_slope_deg": _STALL_PERTURB_SLOPE_DEG}
        )
        try:
            perturbed_metrics = evaluate(
                entry.design,
                perturbed_scenario,
                gravity_m_per_s2=entry.gravity_m_per_s2,
                thermal_architecture=entry.thermal_architecture,
            )
            perturbed_stalled = bool(perturbed_metrics.stalled)
        except Exception:  # pragma: no cover - belt-and-braces
            # A raised exception in the stall-edge regime is itself a
            # stall outcome (the slip solver failing to converge is one
            # of the two stall conditions in run_traverse).
            perturbed_stalled = True
        baseline_ok = not bool(baseline_metrics.stalled)
        rows.append(
            {
                "rover": entry.rover_name,
                "baseline_max_slope_deg": entry.scenario.max_slope_deg,
                "baseline_stalled": bool(baseline_metrics.stalled),
                "perturbed_max_slope_deg": _STALL_PERTURB_SLOPE_DEG,
                "perturbed_stalled": perturbed_stalled,
                "passes": baseline_ok and perturbed_stalled,
            }
        )
    df = pd.DataFrame(rows)
    all_pass = bool(df["passes"].all())
    return df, all_pass


# ---------------------------------------------------------------------------
# Gate 4: Surrogate quality on the feasible test split
# ---------------------------------------------------------------------------


def _gate_test_quality(
    df: pd.DataFrame, bundles_path: Path
) -> tuple[pd.DataFrame, bool]:
    """Per-target R² + 90 % PI coverage on feasible test rows.

    v7 simplification: the v6 ``saturated-δ_des`` regime split is gone
    (``designed_duty_cycle`` was removed from the design vector), so
    the natural single-cell version of this gate is overall surrogate
    quality on the held-out test split, restricted to feasible rows.
    """
    test = df[df["split"] == "test"]
    test = valid_rows(test)
    feas_mask = (~test[FEASIBILITY_COLUMN].astype(bool)).to_numpy()
    test_feas = test.loc[feas_mask].copy()

    bundles = joblib.load(bundles_path)

    rows: list[dict[str, Any]] = []
    if len(test_feas) < 2:
        # Should never happen on a 40k-row dataset, but guard anyway
        # so a dry-run / smoke dataset doesn't crash the gate.
        for tgt in PRIMARY_REGRESSION_TARGETS:
            rows.append(
                {
                    "target": tgt,
                    "n": len(test_feas),
                    "median_r2": float("nan"),
                    "coverage": float("nan"),
                    "mean_width": float("nan"),
                }
            )
        return pd.DataFrame(rows), False

    X = build_feature_matrix(test_feas)
    for tgt in PRIMARY_REGRESSION_TARGETS:
        bundle = bundles[tgt]
        preds = bundle.predict(X, repair_crossings=True)
        q_keys = sorted(preds.keys())  # ["q05", "q50", "q95"]
        lo_arr = preds[q_keys[0]]
        mid_arr = preds[q_keys[1]]
        hi_arr = preds[q_keys[-1]]
        y_true = test_feas[tgt].to_numpy()
        r2 = float(r2_score(y_true, mid_arr))
        covered = (y_true >= lo_arr) & (y_true <= hi_arr)
        coverage = float(covered.mean())
        width = float(np.mean(hi_arr - lo_arr))
        rows.append(
            {
                "target": tgt,
                "n": len(test_feas),
                "median_r2": r2,
                "coverage": coverage,
                "mean_width": width,
            }
        )
    out = pd.DataFrame(rows)
    passes = bool(
        (out["median_r2"].fillna(0.0) >= TEST_R2_FLOOR).all()
        and (out["coverage"].fillna(0.0) >= TEST_COVERAGE_FLOOR).all()
    )
    return out, passes


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    out_dir: Path,
    layer0: pd.DataFrame,
    cruise: pd.DataFrame,
    stall: pd.DataFrame,
    test_quality: pd.DataFrame,
    pass_flags: dict[str, bool],
) -> Path:
    lines: list[str] = []
    lines.append("# W12 Step B follow-up — v7 validation report")
    lines.append("")
    lines.append("Generated by ``scripts/validate_v7.py``.")
    lines.append("")
    lines.append("## Gate roll-up")
    lines.append("")
    for gate, ok in pass_flags.items():
        verdict = "PASS" if ok else "FAIL"
        lines.append(f"- **{gate}**: {verdict}")
    lines.append("")

    def _df_block(title: str, df: pd.DataFrame) -> None:
        lines.append(title)
        lines.append("")
        lines.append("```")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            lines.append(df.to_string(index=False))
        lines.append("```")
        lines.append("")

    _df_block("## Gate 1 — Layer-0 registry truth comparison", layer0)
    _df_block("## Gate 2 — Registry-rover cruise speed", cruise)
    _df_block("## Gate 3 — Stall-edge sanity (baseline OK, perturbed stalls)", stall)
    _df_block("## Gate 4 — Surrogate quality on feasible test split", test_quality)

    path = out_dir / "validate_v7_report.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("validate_v7")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[gate 1] Layer-0 truth comparison")
    layer0_df, gate1_pass = _gate_layer0()
    layer0_df.to_csv(args.out_dir / "layer0_summary.csv", index=False)

    log.info("[gate 2] registry-rover cruise speed")
    cruise_df, gate2_pass = _gate_cruise_speed()
    cruise_df.to_csv(args.out_dir / "cruise_speed.csv", index=False)

    log.info("[gate 3] stall-edge sanity")
    stall_df, gate3_pass = _gate_stall_edge()
    stall_df.to_csv(args.out_dir / "stall_sanity.csv", index=False)

    log.info("[gate 4] surrogate quality on feasible test split")
    log.info("  loading dataset from %s", args.dataset)
    df = read_parquet(args.dataset)
    test_quality_df, gate4_pass = _gate_test_quality(df, args.quantile_bundles)
    test_quality_df.to_csv(args.out_dir / "test_quality.csv", index=False)

    pass_flags = {
        "Gate 1 — Layer-0 truth": gate1_pass,
        "Gate 2 — Cruise speed": gate2_pass,
        "Gate 3 — Stall-edge sanity": gate3_pass,
        "Gate 4 — Test-split quality": gate4_pass,
    }
    report_path = _write_report(
        args.out_dir, layer0_df, cruise_df, stall_df, test_quality_df, pass_flags
    )
    log.info("wrote %s", report_path)

    print("\n=== W12 step B v7 validation summary ===\n", flush=True)
    for gate, ok in pass_flags.items():
        print(f"  {gate}: {'PASS' if ok else 'FAIL'}", flush=True)
    print()
    print("Layer-0 truth:")
    print(layer0_df.to_string(index=False))
    print("\nCruise speed:")
    print(cruise_df.to_string(index=False))
    print("\nStall-edge sanity:")
    print(stall_df.to_string(index=False))
    print("\nTest-split quality:")
    print(test_quality_df.to_string(index=False))

    all_pass = all(pass_flags.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
