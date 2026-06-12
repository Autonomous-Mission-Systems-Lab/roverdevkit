"""Render the four-scenario Pareto-front panel (Fig. 3).

Reproduces ``reports/figures/fig_pareto_fronts.png`` from the committed
evaluator-truth Pareto fronts under ``reports/pareto_fronts/`` (regenerate
those first with ``make pareto-fronts``). The plotting logic itself lives in
:func:`roverdevkit.tradespace.visualize.plot_pareto_fronts`. Part of the
``make figures`` manuscript-figure pipeline.

Usage
-----
::

    python scripts/make_pareto_fronts_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from roverdevkit.tradespace.visualize import plot_pareto_fronts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pareto-dir",
        type=Path,
        default=Path("reports/pareto_fronts"),
        help="Directory holding front_<scenario>.csv files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/figures/fig_pareto_fronts.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = plot_pareto_fronts(args.pareto_dir, args.out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
