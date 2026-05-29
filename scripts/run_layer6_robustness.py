"""Run the Layer-6 ±20 % soil + MassModelParams robustness sweep.

Wraps :func:`roverdevkit.validation.robustness.cross_scenario_robustness_sweep`
with sensible defaults and writes a markdown report plus a flat CSV of
every (scenario × archetype × perturbation) cell so the §8.5 paper
paragraph can cite both the headline ranking-preservation table and
the per-cell drift values.

Usage::

    python scripts/run_layer6_robustness.py \
        --out reports/layer6_robustness/

The default perturbation set is the canonical ±20 % move on:
soil stiffness (``k_c`` and ``k_phi`` jointly), wheel structural
areal density, solar panel areal mass, battery pack specific energy,
and motor specific torque. Enough to cover the dominant
sensitivities flagged by the §8 risk register.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from roverdevkit.validation.robustness import (
    cross_scenario_robustness_sweep,
    format_robustness_report,
)


def _write_entries_csv(summaries: list, path: Path) -> None:
    fieldnames = [
        "perturbation",
        "scenario_name",
        "archetype",
        "baseline_range_km",
        "perturbed_range_km",
        "delta_range_km",
        "baseline_energy_margin_raw_pct",
        "perturbed_energy_margin_raw_pct",
        "delta_energy_margin_raw_pct",
        "baseline_slope_capability_deg",
        "perturbed_slope_capability_deg",
        "delta_slope_capability_deg",
        "baseline_total_mass_kg",
        "perturbed_total_mass_kg",
        "delta_total_mass_kg",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            for entry in s.entries:
                writer.writerow(
                    {
                        "perturbation": entry.perturbation,
                        "scenario_name": entry.scenario_name,
                        "archetype": entry.archetype,
                        "baseline_range_km": entry.baseline_range_km,
                        "perturbed_range_km": entry.perturbed_range_km,
                        "delta_range_km": entry.delta_range_km,
                        "baseline_energy_margin_raw_pct": entry.baseline_energy_margin_raw_pct,
                        "perturbed_energy_margin_raw_pct": entry.perturbed_energy_margin_raw_pct,
                        "delta_energy_margin_raw_pct": entry.delta_energy_margin_raw_pct,
                        "baseline_slope_capability_deg": entry.baseline_slope_capability_deg,
                        "perturbed_slope_capability_deg": entry.perturbed_slope_capability_deg,
                        "delta_slope_capability_deg": entry.delta_slope_capability_deg,
                        "baseline_total_mass_kg": entry.baseline_total_mass_kg,
                        "perturbed_total_mass_kg": entry.perturbed_total_mass_kg,
                        "delta_total_mass_kg": entry.delta_total_mass_kg,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/layer6_robustness"),
        help="Output directory.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[layer6] running cross-scenario robustness sweep ...")
    summaries = cross_scenario_robustness_sweep()
    print(f"[layer6] {len(summaries)} perturbations × {summaries[0].n_cells} cells each")

    report_md = format_robustness_report(summaries)
    md_path = args.out / "layer6_robustness.md"
    md_path.write_text(report_md, encoding="utf-8")
    print(f"[layer6] wrote {md_path}")

    csv_path = args.out / "layer6_entries.csv"
    _write_entries_csv(summaries, csv_path)
    print(f"[layer6] wrote {csv_path}")

    n_passing = sum(1 for s in summaries if s.all_rankings_preserved)
    print(f"[layer6] {n_passing}/{len(summaries)} perturbations preserve all archetype rankings")


if __name__ == "__main__":
    main()
