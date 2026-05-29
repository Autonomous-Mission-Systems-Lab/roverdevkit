import { useMemo } from "react";
import { Play } from "lucide-react";

import { RegistryOverlayPicker } from "@/components/registry-overlay-picker";
import { SweepChart } from "@/components/sweep-chart";
import { SweepConfig } from "@/components/sweep-config";
import { SweepSensitivityHint } from "@/components/sweep-sensitivity-hint";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useRegistry } from "@/hooks/use-registry";
import { useSweep } from "@/hooks/use-sweep";
import { useDesignStore } from "@/store/design-store";
import { useSweepStore } from "@/store/sweep-store";

/**
 * Parametric sweep page.
 *
 * Lets the user vary one or two design-vector fields on a grid and
 * see how a chosen performance metric responds. The base design is
 * imported from the single-design panel's store so a researcher can
 * iterate on a candidate there, then come here to "sweep this thing
 * around it" without re-typing all twelve dimensions.
 *
 * Backend dispatch is handled server-side: the FastAPI route picks
 * the corrected mission evaluator (ground truth) for grids ≤ 200
 * cells in auto mode and the calibrated quantile-XGBoost surrogate
 * (vectorised, fast) above that. The user can also force one
 * backend explicitly when they want to compare or stress-test.
 */
export function ParametricSweep() {
  const baseDesign = useDesignStore((s) => s.design);
  const overlayRovers = useDesignStore((s) => s.overlayRovers);
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const payloadMassOverride = useDesignStore((s) => s.payloadMassOverride);
  const payloadPowerOverride = useDesignStore((s) => s.payloadPowerOverride);

  const target = useSweepStore((s) => s.target);
  const xAxis = useSweepStore((s) => s.xAxis);
  const yAxis = useSweepStore((s) => s.yAxis);
  const backend = useSweepStore((s) => s.backend);

  const sweep = useSweep();

  const { data: registry } = useRegistry();
  const selectedRovers = useMemo(
    () =>
      registry?.rovers.filter((r) => overlayRovers.includes(r.rover_name)) ??
      [],
    [registry, overlayRovers],
  );

  const handleRun = () => {
    // SCHEMA_VERSION v7_1: δ_ops is a true LHS-sampled surrogate
    // input, so the override flows through to both sweep backends
    // (surrogate batch predict + per-cell deterministic evaluator)
    // exactly the same way it does on the Single design tab. The
    // grid is still one-shot — only the constant-across-grid δ_ops
    // changes.
    sweep.mutate({
      target,
      x_axis: xAxis,
      y_axis: yAxis,
      base_design: baseDesign,
      scenario_name: scenarioName,
      backend,
      operational_duty_cycle: opsDutyOverride ?? null,
      // Schema v9: payload is a constant-across-grid mission requirement
      // (only design variables vary on the sweep axes); carry the same
      // override the Single design tab used.
      payload_mass_kg: payloadMassOverride,
      payload_power_w: payloadPowerOverride,
    });
  };

  const errorMessage =
    sweep.error instanceof Error ? sweep.error.message : null;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <Card>
        <CardHeader>
          <CardTitle>Sweep configuration</CardTitle>
          <CardDescription>
            Vary one or two design dimensions on a grid; the rest of the design
            is taken from the Single design tab.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <SweepConfig disabled={sweep.isPending} />
          <RegistryOverlayPicker />
          <Button
            type="button"
            onClick={handleRun}
            disabled={sweep.isPending}
            className="w-full"
            size="lg"
          >
            <Play className="mr-2 h-4 w-4" />
            {sweep.isPending ? "Running sweep…" : "Run sweep"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sweep result</CardTitle>
          <CardDescription>
            {sweep.data
              ? sweepCaption(sweep.data)
              : "Configure the sweep, then click Run."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {errorMessage ? (
            <p className="text-sm text-[var(--color-destructive)]">
              {errorMessage}
            </p>
          ) : sweep.data ? (
            <>
              <SweepChart data={sweep.data} overlayRovers={selectedRovers} />
              <SweepSensitivityHint data={sweep.data} />
            </>
          ) : (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              No sweep has been run yet.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function sweepCaption(data: ReturnType<typeof useSweep>["data"]): string {
  if (!data) return "";
  const backendLabel =
    data.backend_used === "evaluator"
      ? data.used_scm_correction
        ? "corrected evaluator (ground truth)"
        : "evaluator (BW-only fallback)"
      : "surrogate (calibrated)";
  const elapsed =
    data.elapsed_ms < 1000
      ? `${data.elapsed_ms.toFixed(0)} ms`
      : `${(data.elapsed_ms / 1000).toFixed(2)} s`;
  return `${data.n_cells.toLocaleString()} cells via ${backendLabel} · ${elapsed}.`;
}
