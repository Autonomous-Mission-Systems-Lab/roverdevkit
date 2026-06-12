"""Shared matplotlib helpers for static paper figures.

The interactive 3-D Pareto explorer lives in the webapp
(``webapp/frontend/src/pages/pareto-explorer.tsx``) and uses Plotly. This
module is the Python-side home for *static* figures that need to be
reproducible from the ``scripts/make_*_figure.py`` regenerators (driven by
``make figures``): Pareto projections, rediscovery overlays,
surrogate-vs-evaluator accuracy plots, etc.

:func:`set_paper_rcparams` applies the shared style; concrete plotting
functions (e.g. :func:`plot_pareto_fronts`) live here too so the figure
scripts share a single implementation.
"""

from __future__ import annotations

from pathlib import Path

PAPER_FIGURE_DPI = 200
"""Default DPI for raster exports of paper figures."""

CANONICAL_SCENARIO_LABELS: dict[str, str] = {
    "equatorial_mare_traverse": "Mare traverse",
    "polar_prospecting": "Polar prospecting",
    "highland_slope_capability": "Highland slope",
    "crater_rim_survey": "Crater rim survey",
}
"""Canonical Pareto-front scenarios in paper-figure order (slug -> label)."""


def set_paper_rcparams() -> None:
    """Apply the project's standard matplotlib rcParams for paper figures.

    Kept in one place so notebooks and figure-generation scripts produce
    visually consistent output without copy-pasting style blocks. Import
    matplotlib lazily so this module stays import-cheap for callers that
    only need ``PAPER_FIGURE_DPI``.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": PAPER_FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_pareto_fronts(
    pareto_dir: str | Path,
    out_path: str | Path,
    scenarios: dict[str, str] | None = None,
    *,
    show: bool = False,
) -> Path:
    """Render the four-scenario Pareto-front panel (range vs mass, colored by slope).

    Reads the committed ``front_<scenario>.csv`` artifacts produced by
    ``scripts/generate_pareto_fronts.py`` (``make pareto-fronts``) and
    writes a 2x2 PNG. Called by ``scripts/make_pareto_fronts_figure.py``
    (part of the ``make figures`` pipeline).

    Parameters
    ----------
    pareto_dir
        Directory holding ``front_<scenario>.csv`` files.
    out_path
        Destination PNG path; parent directories are created.
    scenarios
        Ordered ``{slug: label}`` mapping. Defaults to
        :data:`CANONICAL_SCENARIO_LABELS`.
    show
        If ``True`` keep the figure open (notebook display); otherwise
        close it after saving (headless script use).

    Returns
    -------
    Path
        The path the figure was written to.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if scenarios is None:
        scenarios = CANONICAL_SCENARIO_LABELS

    set_paper_rcparams()
    pareto_dir = Path(pareto_dir)
    out_path = Path(out_path)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    for ax, (slug, title) in zip(axes.ravel(), scenarios.items()):
        df = pd.read_csv(pareto_dir / f"front_{slug}.csv")
        sc = ax.scatter(
            df["total_mass_kg"],
            df["range_km"],
            c=df["slope_capability_deg"],
            cmap="viridis",
            s=24,
            edgecolor="k",
            linewidth=0.3,
        )
        ax.set_title(f"{title}  (n={len(df)})")
        ax.set_xlabel("total mass (kg)")
        ax.set_ylabel("range (km)")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("slope cap. (deg)")
    fig.suptitle(
        "Scenario-specific Pareto fronts: range vs mass (color = slope capability)"
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
