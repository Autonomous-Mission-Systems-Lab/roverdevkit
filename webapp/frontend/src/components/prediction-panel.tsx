import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { ConstraintDetailsButton } from "@/components/constraint-details-dialog";
import { OutputDetailsButton } from "@/components/output-details-dialog";
import {
  PredictionChart,
  type OverlayPrediction,
} from "@/components/prediction-chart";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type {
  ArchitectureDiagnostic,
  PredictionRow,
  StallDiagnostic,
  ThermalDiagnostic,
} from "@/types/api";
import { TARGET_META } from "@/types/api";

export interface PredictionPanelMeta {
  /** Structured constraint diagnostics from the deterministic evaluator. */
  thermal: ThermalDiagnostic;
  stall: StallDiagnostic;
  architecture: ArchitectureDiagnostic;
}

interface PredictionPanelProps {
  /** Merged rows: evaluator median + (optional) surrogate q05/q95. */
  rows: PredictionRow[] | undefined;
  meta: PredictionPanelMeta | undefined;
  /** Whether either request (evaluate or predict) is in flight. */
  isPending: boolean;
  /** Highest-priority error from the evaluator or surrogate calls. */
  error: Error | null;
  /** Whether the surrogate's PI band is still loading after the evaluator returned. */
  surrogatePending?: boolean;
  overlays?: OverlayPrediction[];
  overlayLoading?: boolean;
  /** Optional banner (e.g. evaluator-only PI warning) above the chart. */
  banner?: ReactNode;
}

/**
 * Right-hand panel: chart + numeric summary table for the latest prediction.
 *
 * The median value is the deterministic output of the analytical mission
 * evaluator (Bekker-Wong); the q05/q95 columns and the chart's blue bars
 * are the surrogate's calibrated 90% prediction interval wrapping that
 * median.
 */
export function PredictionPanel({
  rows,
  meta,
  isPending,
  error,
  surrogatePending = false,
  overlays = [],
  overlayLoading = false,
  banner,
}: PredictionPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Predicted performance</CardTitle>
        <CardDescription>
          Median (♦) is the physics evaluator&rsquo;s deterministic output; the
          blue bar shows the surrogate&rsquo;s calibrated 90% prediction
          interval around it. Selected real rovers appear as coloured circles on
          the chart.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {banner}
        {isPending ? (
          <div className="flex items-center gap-2 text-sm text-[var(--color-muted-foreground)]">
            <Loader2 className="h-4 w-4 animate-spin" />
            Evaluating mission&hellip;
          </div>
        ) : null}

        {error ? (
          <div className="rounded border border-[var(--color-destructive)] bg-[var(--color-destructive)]/5 p-3 text-sm text-[var(--color-destructive)]">
            {error.message}
          </div>
        ) : null}

        {rows && meta ? (
          <>
            <PredictionChart rows={rows} overlays={overlays} />
            {surrogatePending ? (
              <div className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading prediction interval&hellip;
              </div>
            ) : null}
            {overlayLoading ? (
              <div className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading rover comparisons&hellip;
              </div>
            ) : null}
            <div className="overflow-hidden rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-[var(--color-muted)]/40">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Target</th>
                    <th className="px-3 py-2 text-right font-medium">q₀₅</th>
                    <th className="px-3 py-2 text-right font-medium">median</th>
                    <th className="px-3 py-2 text-right font-medium">q₉₅</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const m = TARGET_META[row.target];
                    return (
                      <tr key={row.target} className="border-t">
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-1.5 font-medium">
                            {m.label}
                            <OutputDetailsButton target={row.target} />
                          </div>
                          <div className="text-xs text-[var(--color-muted-foreground)]">
                            {m.description}
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.q05 === null ? "—" : `${fmt(row.q05)} ${m.unit}`}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold tabular-nums">
                          {fmt(row.value)} {m.unit}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {row.q95 === null ? "—" : `${fmt(row.q95)} ${m.unit}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <PanelFooter meta={meta} />
          </>
        ) : isPending ? null : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Configure a rover on the left and click <em>Predict performance</em>{" "}
            to evaluate it.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function PanelFooter({ meta }: { meta: PredictionPanelMeta }) {
  const thermalOk = meta.thermal.survives;
  const driveOk = !meta.stall.stalled;
  const obstacleOk = meta.architecture.obstacle_requirement_met;
  const allOk = thermalOk && driveOk && obstacleOk;
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {allOk ? (
        <>
          <ConstraintChip
            label="thermal survival ok"
            ok
            details={
              <ConstraintDetailsButton
                variant="thermal"
                thermal={meta.thermal}
                stall={meta.stall}
                failed={false}
              />
            }
          />
          <ConstraintChip
            label="rover does not stall"
            ok
            details={
              <ConstraintDetailsButton
                variant="stall"
                thermal={meta.thermal}
                stall={meta.stall}
                failed={false}
              />
            }
          />
          <ConstraintChip
            label={`obstacle margin ${(meta.architecture.obstacle_margin_m * 100).toFixed(0)} cm`}
            ok
            details={null}
          />
        </>
      ) : (
        <>
          {!thermalOk ? (
            <ConstraintChip
              label="thermal survival fails"
              ok={false}
              details={
                <ConstraintDetailsButton
                  variant="thermal"
                  thermal={meta.thermal}
                  stall={meta.stall}
                  failed
                />
              }
            />
          ) : null}
          {!driveOk ? (
            <ConstraintChip
              label="rover stalls"
              ok={false}
              details={
                <ConstraintDetailsButton
                  variant="stall"
                  thermal={meta.thermal}
                  stall={meta.stall}
                  failed
                />
              }
            />
          ) : null}
          {!obstacleOk ? (
            <ConstraintChip
              label="obstacle requirement fails"
              ok={false}
              details={null}
            />
          ) : null}
        </>
      )}
    </div>
  );
}

function ConstraintChip({
  label,
  ok,
  details,
}: {
  label: string;
  ok: boolean;
  details: ReactNode;
}) {
  const cls = ok
    ? "inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2 py-0.5 text-emerald-700"
    : "inline-flex items-center gap-1.5 rounded-full bg-[var(--color-destructive)]/10 px-2 py-0.5 text-[var(--color-destructive)]";
  return (
    <span className={cls}>
      {label}
      {details}
    </span>
  );
}

function fmt(x: number): string {
  if (!Number.isFinite(x)) return "n/a";
  if (Math.abs(x) >= 100) return x.toFixed(1);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toFixed(3);
}
