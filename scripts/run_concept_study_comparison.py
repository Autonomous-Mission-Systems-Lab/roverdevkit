"""Run the Layer-6 concept-study head-to-head comparison and write artifacts.

Drives
:func:`roverdevkit.validation.concept_study_comparison.compare_concept_to_pareto`
for one or more entries in
:data:`roverdevkit.validation.concept_study_comparison.CONCEPT_STUDIES`
(currently just MoonRanger from Kumar et al. i-SAIRAS 2020 #5068) and
emits a paper-ready artifact set under ``--out-dir``.

For each concept study the script writes
``<out-dir>/<concept_lowercase>/`` containing:

- ``comparison.json`` — full :class:`ConceptStudyComparison` dump
  (cited + predicted metrics, dominance counts, design-space
  proximity, per-objective deltas, full Pareto front).
- ``pareto_front.csv`` — one row per Pareto point with design
  fields + objective metrics + ``dominates_concept`` flag.
- ``deltas.csv`` — per-objective table of concept value vs Pareto
  best vs delta vs percent delta.
- ``objective_space.png`` — three 2-D scatter plots of the Pareto
  front in objective space (range vs mass, range vs slope,
  mass vs slope) with the published concept point overlaid.
- ``<concept_lowercase>_report.md`` — human-readable rollup with
  the citation, methodology, headline numbers, and per-figure
  captions suitable for the paper.

Usage
-----
::

    # Default: every registered concept study, evaluator backend
    python scripts/run_concept_study_comparison.py

    # MoonRanger only, surrogate backend, custom budget
    python scripts/run_concept_study_comparison.py \\
        --concept MoonRanger \\
        --backend surrogate \\
        --quantile-bundles reports/surrogate_v9/quantile_bundles.joblib \\
        --population-size 200 --n-generations 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Headless plotting (CI / sandbox) — must run before matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402  (post-MPLBACKEND so matplotlib does not auto-pick a GUI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from roverdevkit.tradespace.optimizer import DEFAULT_OBJECTIVES  # noqa: E402
from roverdevkit.validation.concept_study_comparison import (  # noqa: E402
    CONCEPT_STUDIES,
    ConceptStudyComparison,
    compare_concept_to_pareto,
    get_concept_study,
    list_concept_studies,
)
from roverdevkit.validation.rover_rediscovery import _dominates  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--concept",
        nargs="+",
        default=list_concept_studies(),
        choices=list_concept_studies(),
        help="Concept-study key(s) to compare. Default: all registered.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/concept_study_comparison"),
    )
    p.add_argument(
        "--backend",
        choices=("evaluator", "surrogate"),
        default="evaluator",
    )
    p.add_argument(
        "--quantile-bundles",
        type=Path,
        default=Path("reports/surrogate_v9/quantile_bundles.joblib"),
        help="Required when --backend=surrogate.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--population-size", type=int, default=200)
    p.add_argument("--n-generations", type=int, default=50)
    p.add_argument(
        "--evaluator-eval-cap",
        type=int,
        default=25_000,
        help=(
            "Safety cap on the evaluator backend's pop * gen. Default 25 000 "
            "supports a high-budget 200 * 50 = 10 000 run with headroom."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p.parse_args(argv)


def _comparison_to_json(c: ConceptStudyComparison) -> dict[str, Any]:
    return {
        "rover_name": c.rover_name,
        "scenario_name": c.scenario_name,
        "backend_used": c.backend_used,
        "published_design": c.published_design.model_dump(),
        "published_metrics_cited": {k: float(v) for k, v in c.published_metrics_cited.items()},
        "published_metrics_predicted": {
            k: float(v) for k, v in c.published_metrics_predicted.items()
        },
        "n_pareto_points_dominating_concept": int(c.n_pareto_points_dominating_concept),
        "n_pareto_points_dominated_by_concept": int(c.n_pareto_points_dominated_by_concept),
        "nearest_pareto_index": int(c.nearest_pareto_index),
        "nearest_pareto_design_space_distance": float(c.nearest_pareto_design_space_distance),
        "nearest_pareto_design": c.pareto_designs[c.nearest_pareto_index].model_dump(),
        "nearest_pareto_metrics": {
            k: float(v) for k, v in c.pareto_metrics[c.nearest_pareto_index].items()
        },
        "best_per_objective": {
            target: {k: float(v) for k, v in spec.items()}
            for target, spec in c.best_per_objective.items()
        },
        "pareto_front": [
            {
                "design": d.model_dump(),
                "metrics": {k: float(v) for k, v in m.items()},
            }
            for d, m in zip(c.pareto_designs, c.pareto_metrics, strict=True)
        ],
    }


def _pareto_dataframe(c: ConceptStudyComparison) -> pd.DataFrame:
    """One row per Pareto point with design fields, metrics, and dom flag."""
    rows: list[dict[str, Any]] = []
    for i, (d, m) in enumerate(zip(c.pareto_designs, c.pareto_metrics, strict=True)):
        row: dict[str, Any] = {"pareto_index": i}
        row.update({k: float(v) for k, v in d.model_dump().items()})
        row.update({k: float(v) for k, v in m.items()})
        row["dominates_concept"] = bool(
            _dominates(m, c.published_metrics_predicted, DEFAULT_OBJECTIVES)
        )
        row["normalised_l2_to_concept"] = float(np.nan)  # placeholder, fixed in caller
        rows.append(row)
    return pd.DataFrame(rows)


def _deltas_dataframe(c: ConceptStudyComparison) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, spec in c.best_per_objective.items():
        direction = next(o.direction for o in DEFAULT_OBJECTIVES if o.target == target)
        rows.append(
            {
                "target": target,
                "direction": direction,
                "concept_predicted": spec["concept_value"],
                "concept_cited": c.published_metrics_cited.get(target, float("nan")),
                "pareto_best_value": spec["value"],
                "pareto_best_index": int(spec["pareto_index"]),
                "delta": spec["delta"],
                "percent_delta": spec["percent_delta"],
            }
        )
    return pd.DataFrame(rows)


def _objective_space_scatter(
    c: ConceptStudyComparison, out_path: Path, log: logging.Logger
) -> None:
    """Three-panel 2-D scatter of the Pareto front + concept point.

    Axes pair each objective against each of the other two; the published
    concept point is drawn as a red marker and Pareto points are coloured
    by ``dominates_concept`` so a reviewer can see at a glance which
    Pareto region beats the concept on every metric.
    """
    pareto_metrics = pd.DataFrame(c.pareto_metrics)
    dominates_mask = np.asarray(
        [
            _dominates(m, c.published_metrics_predicted, DEFAULT_OBJECTIVES)
            for m in c.pareto_metrics
        ],
        dtype=bool,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        ("range_km", "total_mass_kg", "Range vs Mass"),
        ("range_km", "slope_capability_deg", "Range vs Slope"),
        ("total_mass_kg", "slope_capability_deg", "Mass vs Slope"),
    ]
    for ax, (xc, yc, title) in zip(axes, panels):
        # Non-dominating Pareto points: faint background.
        non_dom = ~dominates_mask
        ax.scatter(
            pareto_metrics.loc[non_dom, xc],
            pareto_metrics.loc[non_dom, yc],
            s=8,
            alpha=0.35,
            color="#4477AA",
            label="Pareto front",
        )
        # Pareto points that strictly dominate the concept point: highlighted.
        if dominates_mask.any():
            ax.scatter(
                pareto_metrics.loc[dominates_mask, xc],
                pareto_metrics.loc[dominates_mask, yc],
                s=18,
                alpha=0.85,
                color="#228833",
                label="Dominates concept",
            )
        # Concept point.
        ax.scatter(
            [c.published_metrics_predicted[xc]],
            [c.published_metrics_predicted[yc]],
            s=80,
            marker="X",
            color="#CC3311",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label=f"{c.rover_name} (Kumar 2020)",
        )
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.suptitle(
        f"{c.rover_name} concept study vs Pareto front "
        f"(scenario={c.scenario_name}, backend={c.backend_used})",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_path)


def _markdown_report(c: ConceptStudyComparison) -> str:
    """Human-readable rollup of one concept-study comparison."""
    entry = get_concept_study(c.rover_name)
    cited_lines: list[str] = []
    for target in ("range_km", "total_mass_kg", "slope_capability_deg"):
        predicted = c.published_metrics_predicted.get(target)
        cited = c.published_metrics_cited.get(target)
        cited_str = f"{cited:.3f}" if cited is not None else "_(not cited)_"
        predicted_str = f"{predicted:.3f}" if predicted is not None else "_(n/a)_"
        cited_lines.append(f"| {target} | {cited_str} | {predicted_str} |")

    delta_lines: list[str] = []
    for obj in DEFAULT_OBJECTIVES:
        spec = c.best_per_objective[obj.target]
        pct = spec["percent_delta"]
        pct_str = "_(concept = 0)_" if not np.isfinite(pct) else f"{pct:+.1f}%"
        delta_lines.append(
            f"| {obj.target} ({obj.direction}) | "
            f"{spec['concept_value']:.3f} | "
            f"{spec['value']:.3f} | "
            f"{spec['delta']:+.3f} | "
            f"{pct_str} |"
        )

    n_total = len(c.pareto_designs)
    n_dom = c.n_pareto_points_dominating_concept
    nearest_design = c.pareto_designs[c.nearest_pareto_index].model_dump()

    nearest_design_lines: list[str] = []
    for k in sorted(nearest_design):
        v = nearest_design[k]
        if isinstance(v, int):
            nearest_design_lines.append(f"| {k} | {v} |")
        else:
            nearest_design_lines.append(f"| {k} | {v:.4f} |")

    # Detect model-vs-paper discrepancies worth flagging in the writeup.
    findings: list[str] = []
    for target in ("range_km", "total_mass_kg", "slope_capability_deg"):
        cited = c.published_metrics_cited.get(target)
        predicted = c.published_metrics_predicted.get(target)
        if cited is None or predicted is None:
            continue
        if target == "range_km" and predicted < 0.5 and cited >= 1.0:
            findings.append(
                f"- Our corrected evaluator predicts ``range_km = "
                f"{predicted:.2f}`` for the published design under "
                f"`{c.scenario_name}`, well below the paper's "
                f"~{cited:.0f} km headline. The energy margin is "
                f"strongly negative, indicating that under our "
                f"physics the published power architecture does not "
                f"close the kilometre-per-day duty cycle Kumar 2020 "
                f"targeted. This is a *model finding*, not a "
                f"comparison artefact: the same physics is applied "
                f"to every Pareto candidate."
            )
        if target == "total_mass_kg" and predicted > cited * 1.5:
            findings.append(
                f"- Our bottom-up mass model predicts "
                f"``total_mass_kg = {predicted:.1f}`` for the "
                f"published design, ~{predicted / cited:.1f}× the "
                f"paper's cited {cited:.1f} kg total. The cause is "
                f"a schema-semantics gap: Kumar 2020's '{cited:.0f} "
                f"kg' is the total rover mass, whereas our "
                f"``chassis_mass_kg`` field is the structural "
                f"chassis only (the bottom-up model adds wheels, "
                f"motors, solar, battery, avionics, harness, "
                f"thermal, and a 25% margin). The Pareto front "
                f"sees the same mass model, so the head-to-head "
                f"is internally consistent even though the "
                f"absolute mass-axis position of the published "
                f"design is over-stated."
            )
    findings_section = (
        "\n".join(["## Model-vs-paper findings", ""] + findings + [""])
        if findings
        else ""
    )

    is_class_generic = c.scenario_name.endswith("_micro")
    scenario_label = (
        "Class-generic scenario"
        if is_class_generic
        else "Concept paper's mission scenario"
    )
    if is_class_generic:
        scenario_note = (
            "- Leakage controls inherited from Layer-5 rediscovery: the "
            "scenario is a class-generic `*_micro` YAML with δ_ops = 0.10 "
            "(class-neutral) and the per-rover scenario's δ_ops anchor is "
            "not used."
        )
    else:
        scenario_note = (
            f"- Scenario is the rover's own canonical mission YAML "
            f"(`{c.scenario_name}`), representing the mission specs the "
            f"concept team itself targeted. At Layer 6 we measure "
            f"dominance and trade-offs (not closeness to the published "
            f"design vector) so the per-rover scenario's δ_ops anchor "
            f"is a mission requirement, not a leakage signal — see the "
            f"``concept_study_comparison`` module docstring for the "
            f"framing distinction from Layer 5 rediscovery."
        )

    return "\n".join(
        [
            f"# Layer-6 concept-study comparison — {c.rover_name}",
            "",
            f"- Citation: {entry.citation}",
            f"- {scenario_label}: `{c.scenario_name}`",
            f"- NSGA-II backend: `{c.backend_used}`",
            "- Constraint set: "
            + (
                f"mass envelope ≤ {entry.mass_envelope_kg:.1f} kg (class-typical "
                "pre-Phase-A CLPS small-rover slot)"
                if entry.mass_envelope_kg is not None
                else "none (full-class search)"
            ),
            f"- Pareto front size: {n_total}",
            f"- Pareto points strictly dominating the concept on every objective: "
            f"**{n_dom} of {n_total}** ({100 * n_dom / max(1, n_total):.1f}%)",
            f"- Pareto points dominated by the concept: "
            f"{c.n_pareto_points_dominated_by_concept} (well-converged NSGA-II"
            f" produces 0 here)",
            f"- Nearest Pareto point design-space distance "
            f"(normalised L2): **{c.nearest_pareto_design_space_distance:.3f}** "
            f"(uniform-random-pair baseline ≈ 1.22)",
            "",
            "## Cited vs predicted metrics for the concept design",
            "",
            "Cited = headline numbers reported in the concept paper. "
            "Predicted = same design vector evaluated under the analytical "
            "Bekker-Wong physics that our optimiser uses.",
            "",
            "| Target | Concept paper | Our evaluator |",
            "| --- | --- | --- |",
            *cited_lines,
            "",
            findings_section,
            "## Per-objective deltas (concept design vs Pareto best)",
            "",
            "*Pareto best* is the most favourable single Pareto point for "
            "that objective in isolation; it ignores the other two "
            "objectives (one must look at the full front to see the "
            "trade-off). Percent change is undefined when the concept "
            "value is zero.",
            "",
            "| Objective | Concept value | Pareto best | Delta | Percent |",
            "| --- | --- | --- | --- | --- |",
            *delta_lines,
            "",
            "## Nearest Pareto point design",
            "",
            f"This is the Pareto point at normalised-L2 distance "
            f"{c.nearest_pareto_design_space_distance:.3f} from the concept's "
            f"design vector — the one our optimiser would surface as "
            f"the closest alternative.",
            "",
            "| Field | Value |",
            "| --- | --- |",
            *nearest_design_lines,
            "",
            "## Methodology",
            "",
            "- Concept paper's published design vector is evaluated under "
            "the same analytical physics as the Pareto front so the "
            "head-to-head is apples-to-apples regardless of which backend "
            "NSGA-II used.",
            scenario_note,
            "- Mass-envelope constraint replaces the rover's own modelled "
            "mass (Layer 5) with a class-typical pre-Phase-A envelope; the "
            "concept team did not yet have a fixed budget so the optimiser "
            "is given the same envelope they were targeting.",
            "",
            "## Imputation / caveat notes for this entry",
            "",
            entry.notes,
            "",
        ]
    ) + "\n"


def _run_one(
    name: str,
    args: argparse.Namespace,
    bundles: dict[str, Any] | None,
    log: logging.Logger,
) -> tuple[Path, ConceptStudyComparison]:
    entry = get_concept_study(name)
    log.info(
        "comparing concept=%s scenario=%s backend=%s pop=%d gen=%d",
        name,
        entry.scenario_name,
        args.backend,
        args.population_size,
        args.n_generations,
    )
    comparison = compare_concept_to_pareto(
        entry,
        backend=args.backend,
        bundles=bundles,
        population_size=args.population_size,
        n_generations=args.n_generations,
        seed=args.seed,
        evaluator_eval_cap=args.evaluator_eval_cap,
    )

    sub_dir = args.out_dir / name.lower().replace("-", "_")
    sub_dir.mkdir(parents=True, exist_ok=True)

    json_path = sub_dir / "comparison.json"
    json_path.write_text(json.dumps(_comparison_to_json(comparison), indent=2, sort_keys=True))
    log.info("wrote %s", json_path)

    pareto_df = _pareto_dataframe(comparison)
    pareto_path = sub_dir / "pareto_front.csv"
    pareto_df.drop(columns=["normalised_l2_to_concept"]).to_csv(pareto_path, index=False)
    log.info("wrote %s", pareto_path)

    deltas_df = _deltas_dataframe(comparison)
    deltas_path = sub_dir / "deltas.csv"
    deltas_df.to_csv(deltas_path, index=False)
    log.info("wrote %s", deltas_path)

    plot_path = sub_dir / "objective_space.png"
    _objective_space_scatter(comparison, plot_path, log)

    md_path = sub_dir / f"{name.lower().replace('-', '_')}_report.md"
    md_path.write_text(_markdown_report(comparison))
    log.info("wrote %s", md_path)

    return sub_dir, comparison


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    log = logging.getLogger("run_concept_study_comparison")

    bundles = None
    if args.backend == "surrogate":
        if not args.quantile_bundles.exists():
            raise FileNotFoundError(
                f"--backend=surrogate requires quantile bundles; "
                f"file not found: {args.quantile_bundles}"
            )
        bundles = joblib.load(args.quantile_bundles)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rovers_summary: list[dict[str, Any]] = []
    for name in args.concept:
        if name not in CONCEPT_STUDIES:
            raise SystemExit(
                f"unknown concept study {name!r}; known: {list_concept_studies()}"
            )
        sub_dir, comparison = _run_one(name, args, bundles, log)
        rovers_summary.append(
            {
                "concept": name,
                "out_dir": str(sub_dir),
                "scenario": comparison.scenario_name,
                "backend": comparison.backend_used,
                "pareto_size": len(comparison.pareto_designs),
                "n_dominating_concept": comparison.n_pareto_points_dominating_concept,
                "dominating_concept_pct": (
                    100
                    * comparison.n_pareto_points_dominating_concept
                    / max(1, len(comparison.pareto_designs))
                ),
                "nearest_pareto_distance": comparison.nearest_pareto_design_space_distance,
            }
        )

    summary_df = pd.DataFrame(rovers_summary)
    summary_path = args.out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path)

    print()
    print("=== Concept-study comparison summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
