import { AlertCircle, Info } from "lucide-react";

import { DESIGN_BOUNDS, TARGET_META, type SweepResponse } from "@/types/api";

interface SweepSensitivityHintProps {
  data: SweepResponse;
}

/**
 * Inline explanation under the sweep chart that helps the user interpret
 * "flat-looking" results. Drives off the per-axis spread metrics returned
 * by the `/sweep` route. Two distinct modes:
 *
 * 1. Saturation — relative_spread is below SATURATION_THRESHOLD, meaning
 *    the metric varies by less than 1 % of its absolute scale across the
 *    grid. The chart will look flat regardless of color scale; the hint
 *    points the user at this so they don't blame the visualization.
 * 2. Axis-dominance (2-D only) — one axis carries an order of magnitude
 *    more spread than the other. The minor axis still matters but is
 *    being visually masked by the dominant one on the shared color
 *    scale. The hint quantifies the imbalance so the user can decide
 *    whether to drill in on the minor axis on its own.
 *
 * If neither condition holds, the component renders nothing — no hint is
 * better than a noisy one when the chart already speaks for itself.
 */
const SATURATION_RELATIVE_THRESHOLD = 0.01; // 1 %
const AXIS_DOMINANCE_RATIO = 5.0; // major axis ≥ 5× minor axis spread

export function SweepSensitivityHint({ data }: SweepSensitivityHintProps) {
  const { sensitivity, target, x_variable, y_variable } = data;
  const targetMeta = TARGET_META[target];

  // 1. Saturation — applies to both 1-D and 2-D sweeps.
  if (sensitivity.relative_spread < SATURATION_RELATIVE_THRESHOLD) {
    const pct = (sensitivity.relative_spread * 100).toFixed(2);
    return (
      <Hint variant="warning" icon={<AlertCircle className="h-4 w-4" />}>
        <strong>Metric is saturated on this grid.</strong> {targetMeta.label}{" "}
        varies by only {sensitivity.total_spread.toExponential(2)}{" "}
        {targetMeta.unit} ({pct} % of its absolute value) across all cells, so
        the chart looks uniform. Try widening the swept range, switching to a
        more sensitive metric, or using a different scenario.
      </Hint>
    );
  }

  // 2. Axis dominance — only meaningful for 2-D sweeps with both axes
  // contributing nonzero spread. If the minor axis is exactly zero we
  // skip rather than divide.
  if (y_variable !== null && sensitivity.axis_spread_y !== null) {
    const sx = sensitivity.axis_spread_x;
    const sy = sensitivity.axis_spread_y;
    if (sx > 0 && sy > 0) {
      const ratio = Math.max(sx, sy) / Math.min(sx, sy);
      if (ratio >= AXIS_DOMINANCE_RATIO) {
        const dominant = sx >= sy ? x_variable : y_variable;
        const minor = sx >= sy ? y_variable : x_variable;
        const dominantLabel = formatVarLabel(dominant);
        const minorLabel = formatVarLabel(minor);
        return (
          <Hint variant="info" icon={<Info className="h-4 w-4" />}>
            <strong>{dominantLabel} dominates this surface.</strong> Median
            spread along {dominantLabel} is {ratio.toFixed(1)}× larger than
            along {minorLabel}, so {minorLabel}'s effect is real but is visually
            masked by the shared color scale. To see it, run a 1-D sweep over{" "}
            {minorLabel} alone, or freeze {dominantLabel} at the value you care
            about.
          </Hint>
        );
      }
    }
  }

  return null;
}

interface HintProps {
  variant: "warning" | "info";
  icon: React.ReactNode;
  children: React.ReactNode;
}

function Hint({ variant, icon, children }: HintProps) {
  // Soft inline panel rather than a toast: this is a chart caption, not
  // an alert. Tailwind tokens here mirror the rest of the app's surface
  // styling so the hint sits visually under the chart.
  const tone =
    variant === "warning"
      ? "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200"
      : "border-sky-300 bg-sky-50 text-sky-900 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200";
  return (
    <div
      className={`mt-3 flex items-start gap-2 rounded-md border px-3 py-2 text-xs ${tone}`}
      role="note"
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <p className="leading-relaxed">{children}</p>
    </div>
  );
}

function formatVarLabel(variable: string): string {
  const meta = (DESIGN_BOUNDS as Record<string, { label: string }>)[variable];
  return meta?.label ?? variable;
}
