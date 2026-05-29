"""Shared matplotlib helpers for static paper figures.

The interactive 3-D Pareto explorer lives in the webapp
(``webapp/frontend/src/pages/pareto-explorer.tsx``) and uses Plotly. This
module is the Python-side home for *static* figures that need to be
reproducible from a script or notebook: rediscovery overlays, Pareto
projections (range vs mass, mass vs slope, ...), surrogate-vs-evaluator
accuracy plots, partial-dependence plots, etc.

The only thing exported today is :func:`set_paper_rcparams`, which
``notebooks/paper_figures.ipynb`` calls before drawing so that every
static figure shares a consistent font, line-width, and DPI. Concrete
plotting functions land here as the paper figure work progresses.
"""

from __future__ import annotations

PAPER_FIGURE_DPI = 200
"""Default DPI for raster exports of paper figures."""


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
