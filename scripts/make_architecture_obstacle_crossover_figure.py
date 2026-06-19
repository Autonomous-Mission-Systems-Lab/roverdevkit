"""Render architecture obstacle crossover summary figure for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports") / "architecture_obstacle_crossover" / "crossover_summary.csv",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("paper") / "figures" / "fig_architecture_obstacle_crossover.png",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit(f"no rows in {args.summary_csv}")

    if "front_empty" not in df.columns:
        df["front_empty"] = df["n_points"].eq(0)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for scenario_name, group in df.groupby("scenario_name"):
        group = group.sort_values("required_obstacle_height_m")
        x_cm = group["required_obstacle_height_m"] * 100.0
        y_pct = group["frac_rocker_bogie"] * 100.0
        label = scenario_name.replace("_", " ")
        ax.plot(x_cm, y_pct, marker="o", label=label)

        empty = group[group["front_empty"].fillna(group["n_points"].eq(0))]
        if not empty.empty:
            ax.plot(
                empty["required_obstacle_height_m"] * 100.0,
                [0.0] * len(empty),
                linestyle="none",
                marker="x",
                color=ax.lines[-1].get_color(),
                markersize=7,
                markeredgewidth=1.5,
            )

    ax.set_xlabel("Required obstacle height (cm)")
    ax.set_ylabel("Rocker-bogie share of Pareto set (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=200)
    print(f"wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
