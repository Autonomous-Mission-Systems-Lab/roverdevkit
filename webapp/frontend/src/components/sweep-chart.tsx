import type { Data, Layout } from "plotly.js";

import Plot from "@/lib/plotly";
import {
  DESIGN_BOUNDS,
  TARGET_META,
  type RegistryEntrySummary,
  type SweepResponse,
} from "@/types/api";
import { roverColor } from "@/lib/rover-colors";

interface SweepChartProps {
  /** Latest sweep response from `/sweep`. */
  data: SweepResponse;
  /** Registry rovers to overlay (already filtered to "selected"). */
  overlayRovers: RegistryEntrySummary[];
}

/**
 * Plotly chart for parametric sweeps.
 *
 * - 1-D sweeps render as a line of the target metric vs the X
 *   variable, with optional dashed vertical markers at each
 *   selected real rover's value of X.
 * - 2-D sweeps render as a heatmap of the target metric over the
 *   X × Y plane, with optional scatter dots at each selected real
 *   rover's (X, Y) coordinates.
 *
 * Real-rover overlay markers use the same colour palette as the
 * single-design page so a user toggling between tabs sees a
 * consistent visual identity per rover.
 */
export function SweepChart({ data, overlayRovers }: SweepChartProps) {
  const targetMeta = TARGET_META[data.target];
  const xLabel = formatAxisLabel(data.x_variable);
  const yLabel = data.y_variable ? formatAxisLabel(data.y_variable) : null;
  const zLabel = `${targetMeta.label} (${targetMeta.unit})`;

  if (data.y_variable === null) {
    const z = data.z_values as number[];
    const traces: Data[] = [
      {
        type: "scatter",
        mode: "lines+markers",
        x: data.x_values,
        y: z,
        line: { color: "rgb(40, 75, 180)", width: 2 },
        marker: { color: "rgb(40, 75, 180)", size: 6 },
        name: targetMeta.label,
        hovertemplate: `${xLabel}: %{x}<br>${zLabel}: %{y:.3g}<extra></extra>`,
      },
    ];

    // Vertical markers for overlay rovers, at each rover's value of
    // the X variable. Rovers whose X value falls outside the swept
    // range are clipped to the visible window by Plotly automatically.
    overlayRovers.forEach((rover) => {
      const xVal = (rover.design as unknown as Record<string, number>)[
        data.x_variable
      ];
      if (typeof xVal !== "number" || !isFinite(xVal)) return;
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [xVal, xVal],
        y: [Math.min(...z), Math.max(...z)],
        line: { color: roverColor(rover.rover_name), dash: "dash", width: 1.5 },
        name: rover.rover_name,
        hoverinfo: "name",
      });
    });

    const layout: Partial<Layout> = {
      height: 420,
      margin: { l: 70, r: 30, t: 20, b: 60 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { title: { text: xLabel }, zeroline: false },
      yaxis: { title: { text: zLabel }, zeroline: false },
      showlegend: overlayRovers.length > 0,
      legend: { orientation: "h", y: -0.2 },
    };
    return (
      <Plot
        data={traces}
        layout={layout}
        useResizeHandler
        style={{ width: "100%" }}
        config={{ displaylogo: false }}
      />
    );
  }

  // 2-D heatmap
  const z = data.z_values as number[][];
  const traces: Data[] = [
    {
      type: "heatmap",
      x: data.x_values,
      y: data.y_values ?? [],
      z,
      colorscale: "Viridis",
      colorbar: { title: { text: zLabel }, len: 0.8 },
      hovertemplate: `${xLabel}: %{x}<br>${yLabel}: %{y}<br>${zLabel}: %{z:.3g}<extra></extra>`,
    },
  ];

  if (overlayRovers.length > 0) {
    const xs: number[] = [];
    const ys: number[] = [];
    const text: string[] = [];
    const colors: string[] = [];
    overlayRovers.forEach((r) => {
      const xv = (r.design as unknown as Record<string, number>)[
        data.x_variable
      ];
      const yv = (r.design as unknown as Record<string, number>)[
        data.y_variable as string
      ];
      if (
        typeof xv !== "number" ||
        typeof yv !== "number" ||
        !isFinite(xv) ||
        !isFinite(yv)
      ) {
        return;
      }
      xs.push(xv);
      ys.push(yv);
      text.push(r.rover_name);
      colors.push(roverColor(r.rover_name));
    });
    if (xs.length > 0) {
      traces.push({
        type: "scatter",
        mode: "text+markers",
        x: xs,
        y: ys,
        text,
        textposition: "top center",
        textfont: { color: "white", size: 11 },
        marker: {
          color: colors,
          size: 11,
          line: { color: "white", width: 1.5 },
          symbol: "diamond",
        },
        hovertemplate: "%{text}<br>(%{x}, %{y})<extra></extra>",
        name: "Real rovers",
        showlegend: false,
      });
    }
  }

  const layout: Partial<Layout> = {
    height: 480,
    margin: { l: 70, r: 30, t: 20, b: 60 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { title: { text: xLabel } },
    yaxis: { title: { text: yLabel ?? "" } },
    showlegend: false,
  };
  return (
    <Plot
      data={traces}
      layout={layout}
      useResizeHandler
      style={{ width: "100%" }}
      config={{ displaylogo: false }}
    />
  );
}

function formatAxisLabel(variable: string): string {
  const meta = (
    DESIGN_BOUNDS as Record<string, { label: string; unit: string }>
  )[variable];
  if (!meta) return variable;
  return meta.unit ? `${meta.label} (${meta.unit})` : meta.label;
}
