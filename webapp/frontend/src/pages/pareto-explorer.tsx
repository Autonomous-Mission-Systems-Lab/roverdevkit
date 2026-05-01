import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import { Download, Send } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Plot from "@/lib/plotly";
import { useDesignStore } from "@/store/design-store";
import { useParetoStore } from "@/store/pareto-store";
import { useViewStore } from "@/store/view-store";
import type {
  OptimizeObjectiveIn,
  OptimizeParetoPoint,
  PrimaryTarget,
  ScenarioName,
} from "@/types/api";
import { TARGET_META } from "@/types/api";

const DEFAULT_AXES: [PrimaryTarget, PrimaryTarget, PrimaryTarget] = [
  "total_mass_kg",
  "range_km",
  "slope_capability_deg",
];

export function ParetoExplorer() {
  const activeFront = useParetoStore((s) => s.activeFront);
  const setDesign = useDesignStore((s) => s.setDesign);
  const setScenario = useDesignStore((s) => s.setScenario);
  const setView = useViewStore((s) => s.setView);

  const sendPointToDesign = (point: OptimizeParetoPoint) => {
    if (!activeFront) return;
    setDesign(point.design);
    setScenario(activeFront.scenarioName as ScenarioName);
    setView("design");
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Export active front</CardTitle>
          <CardDescription>
            Export the most recent custom Pareto front for notebooks and paper figures.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <ExportButton
            label="CSV"
            filename="pareto_front.csv"
            mime="text/csv"
            content={activeFront ? frontToCsv(activeFront.points) : ""}
            disabled={!activeFront}
          />
          <ExportButton
            label="JSON"
            filename="pareto_front.json"
            mime="application/json"
            content={
              activeFront
                ? JSON.stringify(
                    {
                      label: activeFront.label,
                      scenario_name: activeFront.scenarioName,
                      source: activeFront.source,
                      metadata: activeFront.metadata ?? {},
                      pareto_front: activeFront.points,
                    },
                    null,
                    2,
                  )
                : ""
            }
            disabled={!activeFront}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{activeFront?.label ?? "No front loaded"}</CardTitle>
          <CardDescription>
            {activeFront
              ? `${activeFront.points.length} Pareto points. Click a point or table row to inspect it on the Single design tab.`
              : "Run Compute Pareto or load a canonical front."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeFront ? (
            <div className="space-y-6">
              <ParetoPlot
                objectives={objectivesFromMetadata(activeFront.metadata)}
                points={activeFront.points}
                onSelect={sendPointToDesign}
              />
              <ParetoTable
                points={activeFront.points.slice(0, 12)}
                onSelect={sendPointToDesign}
              />
            </div>
          ) : (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              No Pareto front is active yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ParetoPlot({
  objectives,
  points,
  onSelect,
}: {
  objectives: OptimizeObjectiveIn[];
  points: OptimizeParetoPoint[];
  onSelect: (point: OptimizeParetoPoint) => void;
}) {
  const { xTarget, yTarget, colorTarget } = plotAxes(objectives);
  const trace: Data = {
    type: "scatter",
    mode: "markers",
    x: points.map((p) => p.metrics[xTarget]),
    y: points.map((p) => p.metrics[yTarget]),
    customdata: points.map((_, i) => i),
    marker: {
      size: 9,
      color: colorTarget
        ? points.map((p) => p.metrics[colorTarget])
        : "rgb(40, 75, 180)",
      colorscale: colorTarget ? "Viridis" : undefined,
      colorbar: colorTarget ? { title: { text: axisLabel(colorTarget) } } : undefined,
      line: { color: "white", width: 0.5 },
    },
    text: points.map((p) => hoverText(p)),
    hovertemplate: "%{text}<extra></extra>",
    name: "Pareto front",
  };
  const layout: Partial<Layout> = {
    height: 520,
    margin: { l: 70, r: 30, t: 20, b: 60 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: { text: axisLabel(xTarget) }, zeroline: false },
    yaxis: { title: { text: axisLabel(yTarget) }, zeroline: false },
  };
  return (
    <Plot
      data={[trace]}
      layout={layout}
      useResizeHandler
      style={{ width: "100%" }}
      config={{ displaylogo: false }}
      onClick={(event: Readonly<PlotMouseEvent>) => {
        const raw = event.points?.[0]?.customdata;
        if (typeof raw === "number" && points[raw]) onSelect(points[raw]);
      }}
    />
  );
}

function objectivesFromMetadata(
  metadata: Record<string, unknown> | undefined,
): OptimizeObjectiveIn[] {
  const raw = metadata?.objectives;
  if (!Array.isArray(raw)) return [];
  return raw.filter(isObjective);
}

function isObjective(value: unknown): value is OptimizeObjectiveIn {
  if (!value || typeof value !== "object") return false;
  const row = value as Record<string, unknown>;
  return (
    isPrimaryTarget(row.target) &&
    (row.direction === "min" || row.direction === "max")
  );
}

function isPrimaryTarget(value: unknown): value is PrimaryTarget {
  return (
    value === "range_km" ||
    value === "energy_margin_raw_pct" ||
    value === "slope_capability_deg" ||
    value === "total_mass_kg"
  );
}

function plotAxes(objectives: OptimizeObjectiveIn[]): {
  xTarget: PrimaryTarget;
  yTarget: PrimaryTarget;
  colorTarget: PrimaryTarget | null;
} {
  const selected = uniqueTargets(objectives.map((obj) => obj.target));
  if (selected.length === 0) {
    return {
      xTarget: DEFAULT_AXES[0],
      yTarget: DEFAULT_AXES[1],
      colorTarget: DEFAULT_AXES[2],
    };
  }

  if (selected.length === 1) {
    const yTarget = selected[0];
    const xTarget: PrimaryTarget =
      yTarget === "total_mass_kg" ? "range_km" : "total_mass_kg";
    return { xTarget, yTarget, colorTarget: null };
  }

  const xTarget = selected.includes("total_mass_kg")
    ? "total_mass_kg"
    : selected[0];
  const yTarget = selected.find((target) => target !== xTarget) ?? DEFAULT_AXES[1];
  const colorTarget =
    selected.find((target) => target !== xTarget && target !== yTarget) ??
    null;
  return { xTarget, yTarget, colorTarget };
}

function uniqueTargets(targets: PrimaryTarget[]): PrimaryTarget[] {
  return targets.filter((target, index) => targets.indexOf(target) === index);
}

function ParetoTable({
  points,
  onSelect,
}: {
  points: OptimizeParetoPoint[];
  onSelect: (point: OptimizeParetoPoint) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full min-w-[760px] text-left text-sm">
        <thead className="border-b bg-[var(--color-muted)]/40 text-xs uppercase text-[var(--color-muted-foreground)]">
          <tr>
            <th className="px-3 py-2">Action</th>
            <th className="px-3 py-2">Mass</th>
            <th className="px-3 py-2">Range</th>
            <th className="px-3 py-2">Slope</th>
            <th className="px-3 py-2">Energy</th>
            <th className="px-3 py-2">Design snapshot</th>
          </tr>
        </thead>
        <tbody>
          {points.map((point, index) => (
            <tr key={index} className="border-b last:border-0">
              <td className="px-3 py-2">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => onSelect(point)}
                >
                  <Send className="mr-1 h-3 w-3" />
                  Inspect
                </Button>
              </td>
              <td className="px-3 py-2">
                {formatMetric("total_mass_kg", point.metrics.total_mass_kg)}
              </td>
              <td className="px-3 py-2">
                {formatMetric("range_km", point.metrics.range_km)}
              </td>
              <td className="px-3 py-2">
                {formatMetric(
                  "slope_capability_deg",
                  point.metrics.slope_capability_deg,
                )}
              </td>
              <td className="px-3 py-2">
                {formatMetric(
                  "energy_margin_raw_pct",
                  point.metrics.energy_margin_raw_pct,
                )}
              </td>
              <td className="px-3 py-2 text-xs text-[var(--color-muted-foreground)]">
                R {point.design.wheel_radius_m.toFixed(3)} m ·{" "}
                {point.design.n_wheels} wheels · solar{" "}
                {point.design.solar_area_m2.toFixed(2)} m² · torque{" "}
                {point.design.peak_wheel_torque_nm.toFixed(2)} Nm
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExportButton({
  label,
  filename,
  mime,
  content,
  disabled,
}: {
  label: string;
  filename: string;
  mime: string;
  content: string;
  disabled?: boolean;
}) {
  const href = disabled
    ? undefined
    : URL.createObjectURL(new Blob([content], { type: mime }));
  return (
    <Button asChild variant="outline" disabled={disabled}>
      <a href={href} download={filename}>
        <Download className="mr-2 h-4 w-4" />
        Export {label}
      </a>
    </Button>
  );
}

function hoverText(point: OptimizeParetoPoint): string {
  return [
    `${axisLabel("total_mass_kg")}: ${formatMetric("total_mass_kg", point.metrics.total_mass_kg)}`,
    `${axisLabel("range_km")}: ${formatMetric("range_km", point.metrics.range_km)}`,
    `${axisLabel("slope_capability_deg")}: ${formatMetric("slope_capability_deg", point.metrics.slope_capability_deg)}`,
    `R=${point.design.wheel_radius_m.toFixed(3)} m, W=${point.design.wheel_width_m.toFixed(3)} m`,
    `${point.design.n_wheels} wheels, ${point.design.grouser_count} grousers`,
  ].join("<br>");
}

function axisLabel(target: PrimaryTarget): string {
  const meta = TARGET_META[target];
  return meta.unit ? `${meta.label} (${meta.unit})` : meta.label;
}

function formatMetric(target: PrimaryTarget, value: number): string {
  const unit = TARGET_META[target].unit;
  const digits = Math.abs(value) >= 100 ? 1 : 2;
  return `${value.toFixed(digits)} ${unit}`.trim();
}

function frontToCsv(points: OptimizeParetoPoint[]): string {
  const headers = [
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
    "wheel_radius_m",
    "wheel_width_m",
    "grouser_height_m",
    "grouser_count",
    "n_wheels",
    "chassis_mass_kg",
    "wheelbase_m",
    "solar_area_m2",
    "battery_capacity_wh",
    "avionics_power_w",
    "peak_wheel_torque_nm",
  ];
  const rows = points.map((point) =>
    [
      point.metrics.range_km,
      point.metrics.energy_margin_raw_pct,
      point.metrics.slope_capability_deg,
      point.metrics.total_mass_kg,
      point.design.wheel_radius_m,
      point.design.wheel_width_m,
      point.design.grouser_height_m,
      point.design.grouser_count,
      point.design.n_wheels,
      point.design.chassis_mass_kg,
      point.design.wheelbase_m,
      point.design.solar_area_m2,
      point.design.battery_capacity_wh,
      point.design.avionics_power_w,
      point.design.peak_wheel_torque_nm,
    ].join(","),
  );
  return [headers.join(","), ...rows].join("\n") + "\n";
}
