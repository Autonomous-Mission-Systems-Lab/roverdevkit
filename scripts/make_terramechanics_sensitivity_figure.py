"""Render the terramechanics model-form sensitivity figure.

Overlays the four-scenario Pareto fronts (range vs. mass) produced under a
sweep of the Bekker-Wong shear-stress perturbation
(:func:`roverdevkit.terramechanics.bekker_wong.traction_perturbation`),
showing how far the headline fronts move when the kernel is perturbed by
its own measured drawbar-pull model-form error. Consumes the artifacts
written by ``scripts/run_terramechanics_sensitivity.py``.

Usage
-----
::

    python scripts/run_terramechanics_sensitivity.py   # writes the fronts
    python scripts/make_terramechanics_sensitivity_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from roverdevkit.tradespace.visualize import (  # noqa: E402
    CANONICAL_SCENARIO_LABELS,
    PAPER_FIGURE_DPI,
    set_paper_rcparams,
)

# Nominal shear scale (the unperturbed kernel = the canonical paper fronts).
_NOMINAL_SCALE = 1.00


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sensitivity-dir",
        type=Path,
        default=Path("reports/terramechanics_sensitivity"),
        help="Directory holding front_<scenario>__scale_<s>.csv files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("paper/figures/fig_terramechanics_sensitivity.png"),
    )
    return p.parse_args(argv)


def _scale_label(scale: float, calib: dict[float, float]) -> str:
    """Legend label: shear scale annotated with its median net-DP shift."""
    if scale == _NOMINAL_SCALE:
        return f"{scale:.2f} (nominal)"
    dp = calib.get(round(scale, 2))
    sign = "+" if scale > _NOMINAL_SCALE else "\u2212"
    if dp is not None:
        return f"{scale:.2f} ({sign}{dp:.0f}% DP)"
    return f"{scale:.2f}"


def main(argv: list[str] | None = None) -> int:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import colormaps

    args = _parse_args(argv)
    sens_dir = args.sensitivity_dir

    calib: dict[float, float] = {}
    calib_path = sens_dir / "traction_scale_calibration.csv"
    if calib_path.exists():
        cdf = pd.read_csv(calib_path)
        calib = {
            round(float(r.shear_scale), 2): float(r.median_abs_dp_shift_pct)
            for r in cdf.itertuples()
        }

    set_paper_rcparams()

    # Discover the swept scales from the filenames.
    scales: set[float] = set()
    for path in sens_dir.glob("front_*__scale_*.csv"):
        tag = path.stem.split("__scale_")[-1]
        scales.add(round(float(tag.replace("p", ".")), 2))
    if not scales:
        raise FileNotFoundError(
            f"no front_*__scale_*.csv files in {sens_dir}; "
            "run scripts/run_terramechanics_sensitivity.py first."
        )
    sorted_scales = sorted(scales)

    # Diverging color map centered on the nominal scale: pessimistic
    # traction (scale < 1) blue, optimistic (scale > 1) red.
    span = max(_NOMINAL_SCALE - sorted_scales[0], sorted_scales[-1] - _NOMINAL_SCALE)
    norm = mcolors.Normalize(vmin=_NOMINAL_SCALE - span, vmax=_NOMINAL_SCALE + span)
    cmap = colormaps["coolwarm"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    for ax, (slug, title) in zip(axes.ravel(), CANONICAL_SCENARIO_LABELS.items()):
        for scale in sorted_scales:
            tag = f"{scale:.2f}".replace(".", "p")
            path = sens_dir / f"front_{slug}__scale_{tag}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).sort_values("total_mass_kg")
            # Range-vs-mass is a 2-D projection of a 3-objective
            # (range, mass, slope) front, so the raw points are not
            # monotonic. We draw the *attainment frontier* -- the best
            # range achievable at or below each mass (a cumulative max) --
            # so each scale is one clean curve and the vertical gap
            # between curves is the conclusion band under the perturbation.
            mass = df["total_mass_kg"].to_numpy()
            rng = df["range_km"].cummax().to_numpy()
            is_nominal = scale == _NOMINAL_SCALE
            ax.step(
                mass,
                rng,
                where="post",
                marker="o",
                markersize=3.0 if is_nominal else 2.0,
                linewidth=2.2 if is_nominal else 1.3,
                color="black" if is_nominal else cmap(norm(scale)),
                alpha=1.0 if is_nominal else 0.9,
                zorder=5 if is_nominal else 3,
                label=_scale_label(scale, calib),
            )
        ax.set_title(title)
        ax.set_xlabel("total mass (kg)")
        ax.set_ylabel("max range at \u2264 mass (km)")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="shear scale (= traction model-form perturbation)",
        loc="lower center",
        ncol=len(sorted_scales),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Pareto-front sensitivity to the terramechanics model-form error\n"
        "(range vs. mass; black = unperturbed kernel)"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=PAPER_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
