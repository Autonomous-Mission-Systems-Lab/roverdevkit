"""Propagate the terramechanics kernel's model-form error into the Pareto fronts.

The closed-form Bekker-Wong kernel is the accuracy bottleneck of the
evaluator: against measured single-wheel data its median drawbar-pull
error is ~24-27 % (Section "Terramechanics validation"). This script
asks: *do the headline conclusions
(the four-scenario Pareto fronts and the cross-scenario design rules)
survive a perturbation of the kernel by its own measured model-form
error?*

Method
------
We re-run the **identical** canonical NSGA-II pipeline as
``scripts/generate_pareto_fronts.py`` (same objectives, constraints,
panel orientation, population, generations, and seeds) but wrap each run
in :func:`roverdevkit.terramechanics.bekker_wong.traction_perturbation`,
a multiplicative factor on the mobilised shear stress tau. Because tau
drives the gross tractive effort, the net drawbar pull, driving torque,
and (via the implicit vertical force balance) sinkage all respond
self-consistently.

The shear scale is calibrated to the *measured* drawbar-pull band: on
the digitised single-wheel validation points a scale of +/-0.15 induces
a ~28 % median net-DP shift (matching the 24-27 % measured medians) and
+/-0.30 induces ~55 % (a 2x stress envelope). The calibration is written
to ``traction_scale_calibration.csv`` so the mapping is auditable.

Outputs (under ``--out-dir``, default ``reports/terramechanics_sensitivity``)
-----------------------------------------------------------------------------
- ``front_<scenario>__scale_<s>.csv``   -- one Pareto front per (scenario, scale).
- ``design_rule_robustness.csv``        -- one row per (scenario, scale) with the
  variable medians / bound-pegging fractions / range-mass-slope envelopes that
  back the Section 7.2 design rules.
- ``traction_scale_calibration.csv``    -- shear scale -> median net-DP shift on the
  measured validation points.
- ``manifest.json``                     -- run metadata.

Example
-------
::

    conda run -n roverdevkit --no-capture-output \\
        python scripts/run_terramechanics_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Make the script runnable as ``python scripts/run_terramechanics_sensitivity.py``
# without an editable install or external PYTHONPATH: put the repo root
# (for the ``roverdevkit`` package) and this scripts dir (to reuse the
# canonical Pareto settings) on the path before any project imports.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from roverdevkit.mission.scenarios import list_scenarios, load_scenario  # noqa: E402
from roverdevkit.schema import ScenarioName  # noqa: E402
from roverdevkit.terramechanics.bekker_wong import (  # noqa: E402
    single_wheel_forces,
    traction_perturbation,
)
from roverdevkit.terramechanics.soils import get_soil_parameters  # noqa: E402
from roverdevkit.tradespace.optimizer import (  # noqa: E402
    DEFAULT_OBJECTIVES,
    NSGA2Runner,
    OptimizationConstraint,
)
from roverdevkit.validation.rover_rediscovery import (  # noqa: E402
    _scenario_panel_orientation,
)
from roverdevkit.validation.terramechanics_experiment import (  # noqa: E402
    load_experiment_points,
)

# Reuse the *exact* canonical Pareto settings so the sensitivity fronts
# are comparable to the paper fronts knob-for-knob (only the shear scale
# differs).
from generate_pareto_fronts import (  # noqa: E402
    DEFAULT_EVALUATOR_EVAL_CAP,
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANGE_FLOOR_KM,
    SCENARIO_OVERRIDES,
)

# Shear-stress scales to sweep. 1.00 reproduces the canonical fronts
# bit-for-bit; 0.85/1.15 match the measured +/-~27 % median drawbar-pull
# band; 0.70/1.30 are a 2x (~55 %) stress envelope. See module docstring
# and the emitted calibration CSV.
DEFAULT_SCALES: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15, 1.30)

# Bound-pegging tolerances (fraction of points sitting at a box bound).
_RADIUS_CEIL_M = 0.20
_WIDTH_FLOOR_M = 0.03
_GROUSER_H_CEIL_M = 0.020
_GROUSER_N_CEIL = 24


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports") / "terramechanics_sensitivity",
    )
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCALES),
        help="Shear-stress multipliers to sweep.",
    )
    p.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    p.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    p.add_argument("--seed", type=int, default=12)
    return p.parse_args(argv)


def _scenario_names(raw: list[str] | None) -> list[ScenarioName]:
    allowed = set(list_scenarios())
    values = list_scenarios() if raw is None else raw
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; allowed: {sorted(allowed)}")
    return [name for name in values]  # type: ignore[list-item]


def _calibrate_scales(scales: list[float]) -> pd.DataFrame:
    """Map each shear scale to the median net drawbar-pull shift it induces.

    Evaluated on the digitised single-wheel validation operating points
    (driving slip only) so the scale sweep is anchored to the same
    measured drawbar-pull band quoted in the paper.
    """
    points = [
        p
        for p in load_experiment_points()
        if not math.isnan(p.meas_drawbar_pull_n) and p.slip > 0.0
    ]
    rows: list[dict[str, float | int]] = []
    for scale in scales:
        rel_shifts: list[float] = []
        for pt in points:
            base = single_wheel_forces(pt.wheel, pt.soil, pt.vertical_load_n, pt.slip)
            with traction_perturbation(scale):
                pert = single_wheel_forces(
                    pt.wheel, pt.soil, pt.vertical_load_n, pt.slip
                )
            if base.drawbar_pull_n > 0.0:
                rel_shifts.append(
                    100.0
                    * abs(pert.drawbar_pull_n - base.drawbar_pull_n)
                    / base.drawbar_pull_n
                )
        rows.append(
            {
                "shear_scale": scale,
                "n_validation_points": len(rel_shifts),
                "median_abs_dp_shift_pct": float(np.median(rel_shifts))
                if rel_shifts
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _design_rule_row(
    scenario_name: str, scale: float, front: pd.DataFrame
) -> dict[str, Any]:
    """Summarise the variables that back the Section 7.2 design rules."""
    n = len(front)

    def _frac(mask: "pd.Series[bool]") -> float:
        return float(mask.mean()) if n else math.nan

    if "mobility_architecture" in front.columns:
        rigid_mask = front["mobility_architecture"] == "rigid_4wheel"
    else:
        rigid_mask = front["n_wheels"] == 4

    return {
        "scenario_name": scenario_name,
        "shear_scale": scale,
        "n_points": n,
        # Rule 1: four wheels dominate.
        "frac_four_wheel": _frac(rigid_mask),
        # Rule 2: wheel geometry pegs traction-rich / mass-cheap corner.
        "median_wheel_radius_m": float(front["wheel_radius_m"].median()),
        "frac_radius_at_ceiling": _frac(front["wheel_radius_m"] >= _RADIUS_CEIL_M - 1e-3),
        "median_wheel_width_m": float(front["wheel_width_m"].median()),
        "frac_width_at_floor": _frac(front["wheel_width_m"] <= _WIDTH_FLOOR_M + 1e-3),
        "median_grouser_height_m": float(front["grouser_height_m"].median()),
        "frac_grouser_h_at_ceiling": _frac(
            front["grouser_height_m"] >= _GROUSER_H_CEIL_M - 1e-3
        ),
        "median_grouser_count": float(front["grouser_count"].median()),
        "frac_grouser_n_at_ceiling": _frac(front["grouser_count"] >= _GROUSER_N_CEIL),
        # Rule 3: high-latitude storage vs. array trade.
        "median_battery_capacity_wh": float(front["battery_capacity_wh"].median()),
        "median_solar_area_m2": float(front["solar_area_m2"].median()),
        # Envelope numbers quoted in Section 7.1.
        "range_min_km": float(front["range_km"].min()),
        "range_max_km": float(front["range_km"].max()),
        "mass_min_kg": float(front["total_mass_kg"].min()),
        "mass_max_kg": float(front["total_mass_kg"].max()),
        "slope_median_deg": float(front["slope_capability_deg"].median()),
        "slope_max_deg": float(front["slope_capability_deg"].max()),
        "slope_min_deg": float(front["slope_capability_deg"].min()),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_names(args.scenarios)
    scales = [float(s) for s in args.scales]

    calib = _calibrate_scales(scales)
    calib_path = out_dir / "traction_scale_calibration.csv"
    calib.to_csv(calib_path, index=False)
    print(f"wrote calibration {calib_path}")
    for _, r in calib.iterrows():
        print(
            f"  shear scale {r['shear_scale']:.2f} -> "
            f"median |ΔDP| {r['median_abs_dp_shift_pct']:.1f}% "
            f"(n={int(r['n_validation_points'])})",
            flush=True,
        )

    range_floor = OptimizationConstraint(
        target="range_km", sense="min", value=DEFAULT_RANGE_FLOOR_KM
    )

    robustness_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for i, scenario_name in enumerate(scenarios):
        base_scenario = load_scenario(scenario_name)
        override = SCENARIO_OVERRIDES.get(scenario_name)

        if override is None:
            objectives = DEFAULT_OBJECTIVES
            constraints: tuple[OptimizationConstraint, ...] = (range_floor,)
            scenario = base_scenario
        else:
            objectives = override.objectives
            constraints = (range_floor, *override.extra_constraints)
            scenario = base_scenario
            if override.traverse_distance_m is not None:
                scenario = scenario.model_copy(
                    update={"traverse_distance_m": override.traverse_distance_m}
                )

        soil = get_soil_parameters(scenario.soil_simulant)
        panel_tilt_deg, panel_azimuth_deg = _scenario_panel_orientation(scenario)
        seed = args.seed + i

        for scale in scales:
            t0 = time.perf_counter()
            with traction_perturbation(scale):
                result = NSGA2Runner(
                    scenario,
                    soil,
                    backend="evaluator",
                    objectives=objectives,
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
            if front.empty:
                print(
                    f"{scenario_name} @ scale {scale:.2f}: EMPTY front "
                    f"({elapsed_s:.1f} s)",
                    flush=True,
                )
                continue
            front.insert(0, "shear_scale", scale)
            front.insert(0, "scenario_name", scenario_name)
            scale_tag = f"{scale:.2f}".replace(".", "p")
            front_path = out_dir / f"front_{scenario_name}__scale_{scale_tag}.csv"
            front.to_csv(front_path, index=False)

            robustness_rows.append(_design_rule_row(scenario_name, scale, front))
            manifest.append(
                {
                    "scenario_name": scenario_name,
                    "shear_scale": scale,
                    "pareto_size": len(front),
                    "seed": seed,
                    "population_size": args.population_size,
                    "generations": args.generations,
                    "panel_tilt_deg": panel_tilt_deg,
                    "panel_azimuth_deg": panel_azimuth_deg,
                    "elapsed_s": elapsed_s,
                    "front_csv": str(front_path),
                }
            )
            print(
                f"{scenario_name} @ scale {scale:.2f}: {len(front)} points "
                f"({elapsed_s:.1f} s)",
                flush=True,
            )

    robustness = pd.DataFrame(robustness_rows)
    robustness_path = out_dir / "design_rule_robustness.csv"
    robustness.to_csv(robustness_path, index=False)
    print(f"wrote robustness summary {robustness_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
