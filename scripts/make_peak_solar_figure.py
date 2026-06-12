"""Render the §5.3 de-tuned peak-solar prediction figure.

Reproduces ``reports/figures/fig_validation_peak_solar.png`` from the committed
``reports/power_prediction/summary.csv`` artifact (written by
``scripts/run_power_prediction.py``) so the paper figure is regenerable rather
than a hand-made PNG.

For each flown rover the chart shows:

- the published peak-solar band (grey span) with the published point value;
- the de-tuned clean-array prediction (filled marker) with its
  literature-cell-efficiency sensitivity band (vertical bar) -- this uses a
  single fixed parameter set applied to every rover, no per-rover calibration;
- the de-tuned beginning-of-life clean prediction (open marker) where it
  differs materially, to expose the aging/dust derate the published value of a
  multi-year rover bakes in.

Usage
-----
::

    python scripts/make_peak_solar_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from roverdevkit.tradespace.visualize import set_paper_rcparams

_PUBLISHED_COLOR = "#444444"
_BAND_COLOR = "#cfcfcf"
_PRED_COLOR = "#1f77b4"
_BOL_COLOR = "#d62728"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/power_prediction/summary.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/figures/fig_validation_peak_solar.png"),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = pd.read_csv(args.summary).sort_values("published_w").reset_index(drop=True)

    set_paper_rcparams()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(df)))
    half_w = 0.30

    for i, row in df.iterrows():
        # Published band as a grey span behind everything.
        ax.add_patch(
            plt.Rectangle(
                (i - half_w, row["band_low_w"]),
                2 * half_w,
                row["band_high_w"] - row["band_low_w"],
                facecolor=_BAND_COLOR,
                edgecolor="none",
                zorder=1,
            )
        )
        # Published point value.
        ax.hlines(
            row["published_w"], i - half_w, i + half_w,
            color=_PUBLISHED_COLOR, lw=2.0, zorder=3,
        )
        # De-tuned clean prediction with its cell-efficiency sensitivity band.
        ax.errorbar(
            i, row["predicted_clean_w"],
            yerr=[
                [row["predicted_clean_w"] - row["sensitivity_low_w"]],
                [row["sensitivity_high_w"] - row["predicted_clean_w"]],
            ],
            fmt="o", color=_PRED_COLOR, markersize=7, capsize=4, lw=1.6, zorder=5,
        )
        # BOL clean prediction (open marker) only where it differs from the band.
        bol = row["predicted_bol_w"]
        if bol > row["band_high_w"]:
            ax.scatter(
                i, bol, marker="o", s=55, facecolor="none",
                edgecolor=_BOL_COLOR, linewidths=1.6, zorder=6,
            )
            ax.annotate(
                f"BOL clean: {bol:.0f} W\nimplied derate {row['implied_total_derate']:.2f}",
                xy=(i, bol), xytext=(i + 0.12, bol),
                va="center", ha="left", fontsize=8, color=_BOL_COLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["rover_name"])
    ax.set_ylabel("peak solar power (W)")
    ax.set_xlim(-0.6, len(df) - 0.4 + 0.9)
    ax.set_ylim(0, float(df["predicted_bol_w"].max()) * 1.15)
    ax.set_title("Fixed-parameter peak-solar prediction vs published band (no per-rover tuning)")

    legend_handles = [
        Patch(facecolor=_BAND_COLOR, label="published band"),
        Line2D([0], [0], color=_PUBLISHED_COLOR, lw=2.0, label="published value"),
        Line2D(
            [0], [0], marker="o", linestyle="none", color=_PRED_COLOR, markersize=7,
            label="fixed-parameter clean prediction (cell-eff. sensitivity bar)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=_BOL_COLOR, markersize=8, label="fixed-parameter BOL clean prediction",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
