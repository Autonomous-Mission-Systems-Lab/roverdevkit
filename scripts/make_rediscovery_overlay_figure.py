"""Render the flown-rover rediscovery mass--range overlay (Fig. 5).

Reproduces ``paper/figures/fig_rediscovery_overlay.png`` from the committed
rediscovery artifacts under
``reports/rediscovery_loo_evaluator/`` (regenerate those with
``python scripts/run_rediscovery_loo.py --all``). For each flown rover the
panel overlays its real design point and the nearest Pareto design on the
optimizer's front in the (total mass, range) plane.

Usage
-----
::

    python scripts/make_rediscovery_overlay_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roverdevkit.tradespace.visualize import set_paper_rcparams

# Flown rovers shown in the overlay: artifact slug -> panel label.
FLOWN_ROVERS: dict[str, str] = {
    "pragyan": "Pragyan (polar)",
    "yutu_2": "Yutu-2 (mare)",
}

_FRONT_COLOR = "#bbbbbb"
_NEAREST_COLOR = "#1b7837"
_ROVER_COLOR = "#d6604d"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--rediscovery-dir",
        type=Path,
        default=Path("reports/rediscovery_loo_evaluator"),
        help="Directory holding <rover>.json rediscovery artifacts.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/fig_rediscovery_overlay.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    set_paper_rcparams()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (slug, label) in zip(axes, FLOWN_ROVERS.items()):
        d = json.loads((args.rediscovery_dir / f"{slug}.json").read_text())
        front = d["pareto_front"]
        mass = [p["metrics"]["total_mass_kg"] for p in front]
        rng = [p["metrics"]["range_km"] for p in front]
        ax.scatter(mass, rng, s=14, color=_FRONT_COLOR, label="Pareto front")

        nm = d["nearest_pareto_metrics"]
        rm = d["rover_metrics_under_generic_scenario"]
        ax.scatter(
            [nm["total_mass_kg"]], [nm["range_km"]],
            color=_NEAREST_COLOR, s=80, marker="o", zorder=5, label="nearest design",
        )
        ax.scatter(
            [rm["total_mass_kg"]], [rm["range_km"]],
            color=_ROVER_COLOR, s=130, marker="*", zorder=6, label="real rover",
        )
        ax.set_title(f"{label}\ndesign-space distance = {d['design_space_distance']:.2f}")
        ax.set_xlabel("total mass (kg)")
        ax.set_ylabel("range (km)")
        ax.legend(loc="best")

    fig.suptitle("Rediscovery overlay: real rover vs nearest Pareto design")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
