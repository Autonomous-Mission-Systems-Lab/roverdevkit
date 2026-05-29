import { useMemo, useState } from "react";
import type { Data, Layout, PlotMouseEvent } from "plotly.js";
import { Send } from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import Plot from "@/lib/plotly";
import {
  useRediscoveryDetail,
  useRediscoveryList,
} from "@/hooks/use-rediscovery";
import { useDesignStore } from "@/store/design-store";
import { useViewStore } from "@/store/view-store";
import type {
  DesignVector,
  PrimaryTarget,
  RediscoveryBackend,
  RediscoveryDetail,
  RediscoveryParetoPoint,
  RediscoverySummary,
  ScenarioName,
} from "@/types/api";
import { TARGET_META } from "@/types/api";

const PLOT_AXES: { x: PrimaryTarget; y: PrimaryTarget; color: PrimaryTarget } = {
  x: "total_mass_kg",
  y: "range_km",
  color: "slope_capability_deg",
};

/**
 * Maps the rediscovery class-generic `*_micro` scenario name onto the
 * closest user-facing canonical scenario in the design panel.
 *
 * The class-generic scenarios live in `roverdevkit/mission/configs/`
 * but are not in the design panel's `ScenarioName` literal because
 * they are leakage-controlled siblings of the canonical four — same
 * latitude / terrain / sun geometry character, different δ_ops
 * anchor and mass-only budget. When the user clicks "send to design
 * panel" we drop them on the closest canonical scenario so the
 * design they just imported lands on a meaningful follow-up rather
 * than the design-panel default.
 */
const SCENARIO_MAP: Record<string, ScenarioName> = {
  polar_micro: "polar_prospecting",
  mare_micro: "equatorial_mare_traverse",
  highland_micro: "highland_slope_capability",
  crater_rim_micro: "crater_rim_survey",
};

export function RediscoveryValidation() {
  const [backend, setBackend] = useState<RediscoveryBackend>("evaluator");
  // `userSelectedSlug` records explicit clicks; `null` means "fall
  // back to the default selection". Deriving the active slug rather
  // than syncing it via `useEffect` avoids the cascading-render
  // pattern that React 19 + the ESLint hooks plugin flag.
  const [userSelectedSlug, setUserSelectedSlug] = useState<string | null>(null);

  const list = useRediscoveryList(backend);
  const summaries = useMemo(() => list.data?.rovers ?? [], [list.data]);

  const activeSlug = useMemo(() => {
    if (userSelectedSlug) {
      const stillThere = summaries.some((s) => s.slug === userSelectedSlug);
      if (stillThere) return userSelectedSlug;
    }
    if (summaries.length === 0) return null;
    return (summaries.find((r) => r.is_flown) ?? summaries[0]).slug;
  }, [summaries, userSelectedSlug]);

  const detail = useRediscoveryDetail(activeSlug, backend);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Rediscovery validation (Layer 5)</CardTitle>
          <CardDescription>
            For each registry rover the leakage-controlled NSGA-II optimiser
            ran a Leave-One-Out search under a class-generic{" "}
            <span className="font-mono">*_micro</span> scenario with a mass-only
            budget. The figure overlays the rover&apos;s published design point
            on the optimiser&apos;s Pareto front so you can see at a glance how
            close the optimiser places it. Flown rovers are labelled separately
            from design-target registry cases. Source artifacts:{" "}
            <span className="font-mono">
              reports/rediscovery_loo_{backend}/
            </span>
            .
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-4">
          <div className="space-y-1">
            <label
              htmlFor="rediscovery-backend"
              className="text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]"
            >
              Backend
            </label>
            <Select
              value={backend}
              onValueChange={(v) => setBackend(v as RediscoveryBackend)}
            >
              <SelectTrigger
                id="rediscovery-backend"
                className="w-[260px]"
                aria-label="Select rediscovery backend"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="evaluator">
                  Evaluator (corrected physics, paper-grade)
                </SelectItem>
                <SelectItem value="surrogate">
                  Surrogate v8 (faster, horizontal-panel reference)
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          <p className="max-w-md text-xs text-[var(--color-muted-foreground)]">
            The evaluator backend is the source of truth. The v8 surrogate is
            offered as a wall-clock reference; per the 2026-05-28 panel-tilt
            fix its polar predictions are still horizontal-panel based.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Registry rovers ({summaries.length})</CardTitle>
          <CardDescription>
            Click a rover to load its Pareto front below. Flown rovers
            (Pragyan, Yutu-2) are highlighted; the rest are well-spec&apos;d
            design-target lunar micro-rovers added for Layer-5 OOD coverage.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {list.isPending ? (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Loading rediscovery summaries…
            </p>
          ) : list.isError ? (
            <p className="text-sm text-[var(--color-destructive)]">
              {list.error instanceof Error
                ? list.error.message
                : "Failed to load rediscovery summaries."}
            </p>
          ) : (
            <RoverGrid
              summaries={summaries}
              activeSlug={activeSlug}
              onSelect={setUserSelectedSlug}
            />
          )}
        </CardContent>
      </Card>

      {activeSlug ? (
        <RoverDetailCard
          slug={activeSlug}
          isPending={detail.isPending}
          isError={detail.isError}
          error={detail.error}
          data={detail.data}
        />
      ) : null}
    </div>
  );
}

function RoverGrid({
  summaries,
  activeSlug,
  onSelect,
}: {
  summaries: RediscoverySummary[];
  activeSlug: string | null;
  onSelect: (slug: string) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {summaries.map((rover) => {
        const active = rover.slug === activeSlug;
        return (
          <button
            type="button"
            key={rover.slug}
            onClick={() => onSelect(rover.slug)}
            aria-pressed={active}
            className={cn(
              "group flex flex-col items-start gap-2 rounded-md border p-3 text-left text-sm transition",
              active
                ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10"
                : "hover:border-[var(--color-accent)] hover:bg-[var(--color-muted)]/40",
            )}
          >
            <div className="flex w-full items-start justify-between gap-2">
              <span className="font-medium">{rover.rover_name}</span>
              <RoverTierBadge isFlown={rover.is_flown} />
            </div>
            <span className="text-xs text-[var(--color-muted-foreground)]">
              <span className="font-mono">{rover.class_generic_scenario}</span>
              {" · "}front: {rover.pareto_front_size} pts
            </span>
            <div className="flex w-full flex-wrap gap-x-3 gap-y-1 text-xs">
              <Stat
                label="distance"
                value={rover.design_space_distance.toFixed(3)}
                emphasis={rover.design_space_distance < 0.4}
              />
              <Stat
                label="dominated"
                value={rover.pareto_dominated ? "yes" : "no"}
                emphasis={!rover.pareto_dominated}
              />
              <Stat
                label="mass budget"
                value={`${rover.mass_budget_kg.toFixed(1)} kg`}
              />
            </div>
          </button>
        );
      })}
    </div>
  );
}

function RoverDetailCard({
  slug,
  isPending,
  isError,
  error,
  data,
}: {
  slug: string;
  isPending: boolean;
  isError: boolean;
  error: unknown;
  data: RediscoveryDetail | undefined;
}) {
  const setDesign = useDesignStore((s) => s.setDesign);
  const setScenario = useDesignStore((s) => s.setScenario);
  const setView = useViewStore((s) => s.setView);

  const sendToDesignPanel = (design: DesignVector, scenario: string) => {
    setDesign(design);
    const mapped = SCENARIO_MAP[scenario];
    if (mapped) setScenario(mapped);
    setView("design");
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {data?.rover_name ?? slug}
          {data ? (
            <span className="ml-2 text-sm text-[var(--color-muted-foreground)]">
              (
              <span className="font-mono">{data.class_generic_scenario}</span>
              {", "}
              <span>distance {data.design_space_distance.toFixed(3)}</span>
              {", "}
              <span>
                {data.pareto_dominated ? "dominated" : "non-dominated"}
              </span>
              )
            </span>
          ) : null}
        </CardTitle>
        <CardDescription>
          Rover&apos;s published design point overlaid on the NSGA-II Pareto
          front under its class-generic scenario. Click any front point or
          the rover&apos;s nearest-Pareto pick to inspect that design on the
          Current design tab.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isPending ? (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Loading rover detail…
          </p>
        ) : isError ? (
          <p className="text-sm text-[var(--color-destructive)]">
            {error instanceof Error
              ? error.message
              : "Failed to load rover detail."}
          </p>
        ) : data ? (
          <div className="space-y-6">
            <RediscoveryPlot data={data} onPickPoint={sendToDesignPanel} />
            <RoverMetricsTable data={data} />
            <NearestParetoCard
              data={data}
              onSendNearest={() =>
                sendToDesignPanel(
                  data.nearest_pareto_design,
                  data.class_generic_scenario,
                )
              }
              onSendRover={() =>
                sendToDesignPanel(
                  data.rover_design,
                  data.class_generic_scenario,
                )
              }
            />
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function RediscoveryPlot({
  data,
  onPickPoint,
}: {
  data: RediscoveryDetail;
  onPickPoint: (design: DesignVector, scenario: string) => void;
}) {
  const front = data.pareto_front;
  const xKey = PLOT_AXES.x;
  const yKey = PLOT_AXES.y;
  const colorKey = PLOT_AXES.color;

  const traces: Data[] = useMemo(() => {
    const frontTrace: Data = {
      type: "scatter",
      mode: "markers",
      name: "Pareto front",
      x: front.map((p) => p.metrics[xKey]),
      y: front.map((p) => p.metrics[yKey]),
      customdata: front.map((_, i) => i),
      marker: {
        size: 7,
        color: front.map((p) => p.metrics[colorKey] ?? 0),
        colorscale: "Viridis",
        colorbar: { title: { text: axisLabel(colorKey) } },
        line: { color: "white", width: 0.5 },
      },
      text: front.map((p) => paretoHover(p)),
      hovertemplate: "%{text}<extra></extra>",
    };
    const roverMetrics = data.rover_metrics_under_generic_scenario;
    const roverTrace: Data = {
      type: "scatter",
      mode: "markers",
      name: data.is_flown ? `${data.rover_name} (flown)` : data.rover_name,
      x: [roverMetrics[xKey] ?? 0],
      y: [roverMetrics[yKey] ?? 0],
      marker: {
        size: 18,
        symbol: data.is_flown ? "star" : "diamond",
        color: data.is_flown ? "rgb(220, 38, 38)" : "rgb(245, 158, 11)",
        line: { color: "black", width: 1.5 },
      },
      text: [roverHover(data.rover_name, roverMetrics)],
      hovertemplate: "%{text}<extra></extra>",
    };
    const nearestTrace: Data = {
      type: "scatter",
      mode: "markers",
      name: "Nearest Pareto pick",
      x: [data.nearest_pareto_metrics[xKey] ?? 0],
      y: [data.nearest_pareto_metrics[yKey] ?? 0],
      marker: {
        size: 14,
        symbol: "circle-open",
        color: "rgb(15, 118, 110)",
        line: { color: "rgb(15, 118, 110)", width: 2 },
      },
      text: [paretoHover({
        design: data.nearest_pareto_design,
        metrics: data.nearest_pareto_metrics,
      })],
      hovertemplate: "%{text}<extra></extra>",
    };
    return [frontTrace, roverTrace, nearestTrace];
  }, [colorKey, data, front, xKey, yKey]);

  const layout: Partial<Layout> = {
    height: 520,
    margin: { l: 70, r: 30, t: 30, b: 60 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    legend: { orientation: "h", y: -0.2 },
    xaxis: { title: { text: axisLabel(xKey) }, zeroline: false },
    yaxis: { title: { text: axisLabel(yKey) }, zeroline: false },
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      useResizeHandler
      style={{ width: "100%" }}
      config={{ displaylogo: false }}
      onClick={(event: Readonly<PlotMouseEvent>) => {
        const first = event.points?.[0];
        if (!first) return;
        const idx = first.customdata;
        if (typeof idx === "number" && front[idx]) {
          onPickPoint(front[idx].design, data.class_generic_scenario);
        }
      }}
    />
  );
}

function RoverMetricsTable({ data }: { data: RediscoveryDetail }) {
  const targets: PrimaryTarget[] = [
    "range_km",
    "energy_margin_raw_pct",
    "slope_capability_deg",
    "total_mass_kg",
  ];
  const rover = data.rover_metrics_under_generic_scenario;
  const nearest = data.nearest_pareto_metrics;
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="border-b bg-[var(--color-muted)]/40 text-xs uppercase text-[var(--color-muted-foreground)]">
          <tr>
            <th className="px-3 py-2">Metric</th>
            <th className="px-3 py-2">{data.rover_name}</th>
            <th className="px-3 py-2">Nearest Pareto</th>
            <th className="px-3 py-2">Δ vs nearest</th>
          </tr>
        </thead>
        <tbody>
          {targets.map((t) => {
            const r = rover[t];
            const n = nearest[t];
            const hasBoth = Number.isFinite(r) && Number.isFinite(n);
            const delta = hasBoth ? r - n : NaN;
            return (
              <tr key={t} className="border-b last:border-0">
                <td className="px-3 py-2">{TARGET_META[t].label}</td>
                <td className="px-3 py-2">{formatTarget(t, r)}</td>
                <td className="px-3 py-2">{formatTarget(t, n)}</td>
                <td className="px-3 py-2 text-[var(--color-muted-foreground)]">
                  {Number.isFinite(delta) ? formatTarget(t, delta) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function NearestParetoCard({
  data,
  onSendNearest,
  onSendRover,
}: {
  data: RediscoveryDetail;
  onSendNearest: () => void;
  onSendRover: () => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="rounded-md border p-3 text-sm">
        <div className="mb-2 flex items-start justify-between gap-2">
          <div>
            <h3 className="font-medium">Real rover design</h3>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              Published design vector from the registry.
            </p>
          </div>
          <Button type="button" size="sm" variant="outline" onClick={onSendRover}>
            <Send className="mr-1 h-3 w-3" />
            Send to design
          </Button>
        </div>
        <DesignSnapshot design={data.rover_design} />
      </div>
      <div className="rounded-md border p-3 text-sm">
        <div className="mb-2 flex items-start justify-between gap-2">
          <div>
            <h3 className="font-medium">
              Nearest Pareto pick (#{data.nearest_pareto_index})
            </h3>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              Optimiser&apos;s closest point to the rover in design space.
            </p>
          </div>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={onSendNearest}
          >
            <Send className="mr-1 h-3 w-3" />
            Send to design
          </Button>
        </div>
        <DesignSnapshot design={data.nearest_pareto_design} />
      </div>
    </div>
  );
}

function DesignSnapshot({ design }: { design: DesignVector }) {
  return (
    <ul className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs">
      <li>R = {design.wheel_radius_m.toFixed(3)} m</li>
      <li>W = {design.wheel_width_m.toFixed(3)} m</li>
      <li>{design.n_wheels} wheels</li>
      <li>{design.grouser_count} grousers</li>
      <li>chassis {design.chassis_mass_kg.toFixed(2)} kg</li>
      <li>solar {design.solar_area_m2.toFixed(2)} m²</li>
      <li>battery {design.battery_capacity_wh.toFixed(0)} Wh</li>
      <li>τ {design.peak_wheel_torque_nm.toFixed(2)} Nm</li>
    </ul>
  );
}

function RoverTierBadge({ isFlown }: { isFlown: boolean }) {
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        isFlown
          ? "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200"
          : "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
      )}
    >
      {isFlown ? "Flown" : "Design target"}
    </span>
  );
}

function Stat({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <span>
      <span className="text-[var(--color-muted-foreground)]">{label}: </span>
      <span className={cn("font-mono", emphasis && "font-semibold")}>{value}</span>
    </span>
  );
}

function paretoHover(point: RediscoveryParetoPoint): string {
  const m = point.metrics;
  return [
    `${TARGET_META.total_mass_kg.label}: ${formatTarget("total_mass_kg", m.total_mass_kg)}`,
    `${TARGET_META.range_km.label}: ${formatTarget("range_km", m.range_km)}`,
    `${TARGET_META.slope_capability_deg.label}: ${formatTarget("slope_capability_deg", m.slope_capability_deg)}`,
    `R=${point.design.wheel_radius_m.toFixed(3)} m, ${point.design.n_wheels} wheels`,
    `solar ${point.design.solar_area_m2.toFixed(2)} m², τ ${point.design.peak_wheel_torque_nm.toFixed(2)} Nm`,
    "Click to inspect on Current design tab.",
  ].join("<br>");
}

function roverHover(name: string, metrics: Record<string, number>): string {
  return [
    `<b>${name}</b> (rover_metrics_under_generic_scenario)`,
    `${TARGET_META.total_mass_kg.label}: ${formatTarget("total_mass_kg", metrics.total_mass_kg ?? 0)}`,
    `${TARGET_META.range_km.label}: ${formatTarget("range_km", metrics.range_km ?? 0)}`,
    `${TARGET_META.slope_capability_deg.label}: ${formatTarget("slope_capability_deg", metrics.slope_capability_deg ?? 0)}`,
    `${TARGET_META.energy_margin_raw_pct.label}: ${formatTarget("energy_margin_raw_pct", metrics.energy_margin_raw_pct ?? 0)}`,
  ].join("<br>");
}

function axisLabel(target: PrimaryTarget): string {
  const meta = TARGET_META[target];
  return meta.unit ? `${meta.label} (${meta.unit})` : meta.label;
}

function formatTarget(target: PrimaryTarget, value: number): string {
  if (!Number.isFinite(value)) return "—";
  const unit = TARGET_META[target].unit;
  const digits = Math.abs(value) >= 100 ? 1 : 2;
  return `${value.toFixed(digits)} ${unit}`.trim();
}
