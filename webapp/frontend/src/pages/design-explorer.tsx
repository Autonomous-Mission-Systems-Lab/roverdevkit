import { useMemo } from "react";
import { Play } from "lucide-react";

import { DesignForm, type DesignFormTicks } from "@/components/design-form";
import { MissionInputsPanel } from "@/components/mission-inputs-panel";
import { NoPiBanner } from "@/components/no-pi-banner";
import {
  PredictionPanel,
  type PredictionPanelMeta,
} from "@/components/prediction-panel";
import type { OverlayPrediction } from "@/components/prediction-chart";
import { RegistryOverlayPicker } from "@/components/registry-overlay-picker";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useEvaluate, useRegistryEvaluations } from "@/hooks/use-evaluate";
import { usePredict } from "@/hooks/use-predict";
import { useRegistry } from "@/hooks/use-registry";
import { roverColor } from "@/lib/rover-colors";
import { useDesignStore } from "@/store/design-store";
import {
  DESIGN_BOUNDS,
  PRIMARY_REGRESSION_TARGET_ORDER,
  type DesignVector,
  type PredictionRow,
  type PrimaryTarget,
} from "@/types/api";

/**
 * Single-design panel: mission inputs and rover comparison across the
 * top, then design inputs on the left and predicted performance on the
 * right.
 *
 * The chart's median diamond is the corrected mission evaluator's
 * deterministic output; the surrogate's quantile heads supply the
 * blue 90% prediction-interval band wrapping that median. Overlays
 * use the evaluator too so candidate-vs-flown comparisons are
 * apples-to-apples ground truth.
 */
export function DesignExplorer() {
  const design = useDesignStore((s) => s.design);
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const payloadMassOverride = useDesignStore((s) => s.payloadMassOverride);
  const payloadPowerOverride = useDesignStore((s) => s.payloadPowerOverride);
  const overlayRovers = useDesignStore((s) => s.overlayRovers);

  const evaluate = useEvaluate();
  const predict = usePredict();

  const { data: registry } = useRegistry();

  const selectedRovers = useMemo(
    () =>
      registry?.rovers.filter((r) => overlayRovers.includes(r.rover_name)) ??
      [],
    [registry, overlayRovers],
  );

  // Only run overlay evaluations once the candidate has at least one
  // result — there's no chart to overlay onto before that, and we
  // don't want surprise traffic on first paint.
  const overlayInputs = evaluate.data
    ? selectedRovers.map((r) => ({
        rover_name: r.rover_name,
        design: r.design,
      }))
    : [];

  // Carry the same mission requirements (payload mass/power) onto the
  // real-rover overlays so candidate-vs-flown comparisons share an
  // identical mission budget on the chart.
  const overlayQueries = useRegistryEvaluations(overlayInputs, scenarioName, {
    payload_mass_kg: payloadMassOverride,
    payload_power_w: payloadPowerOverride,
  });

  const overlays: OverlayPrediction[] = overlayInputs
    .map((input, idx) => {
      const result = overlayQueries.results[idx];
      if (!result?.data) return null;
      return {
        rover_name: input.rover_name,
        color: roverColor(input.rover_name),
        metrics: result.data.metrics,
      } satisfies OverlayPrediction;
    })
    .filter((o): o is OverlayPrediction => o !== null);

  // Slider tick data: one entry per design-vector field, populated
  // with the selected rovers' values. The form already knows how to
  // render these as colour-coded marks above each slider track.
  const formTicks: DesignFormTicks = useMemo(() => {
    const result: DesignFormTicks = {};
    const fields = Object.keys(DESIGN_BOUNDS) as (keyof DesignVector)[];
    for (const field of fields) {
      result[field] = selectedRovers.map((r) => ({
        rover_name: r.rover_name,
        value: r.design[field] as number,
        color: roverColor(r.rover_name),
      }));
    }
    return result;
  }, [selectedRovers]);

  const rows = useMemo<PredictionRow[] | undefined>(() => {
    if (!evaluate.data) return undefined;
    const evalByTarget = new Map(
      evaluate.data.metrics.map((m) => [m.target, m.value]),
    );
    // Schema v7_1 (v7_1 schema follow-on): δ_ops is now an LHS feature,
    // so the surrogate keeps its calibrated PIs across the whole
    // override range. ``mode`` is always ``"surrogate"`` from the
    // live route, but we still gate on it so the band is suppressed
    // if a future evaluator-only fallback is added.
    const showSurrogateBand = predict.data && predict.data.mode === "surrogate";
    const surrByTarget = new Map(
      showSurrogateBand
        ? predict.data!.predictions.map((p) => [
            p.target,
            { q05: p.q05, q95: p.q95 },
          ])
        : [],
    );
    return PRIMARY_REGRESSION_TARGET_ORDER.map((target: PrimaryTarget) => {
      const value = evalByTarget.get(target);
      if (value === undefined) return null;
      const surr = surrByTarget.get(target);
      return {
        target,
        value,
        q05: surr ? surr.q05 : null,
        q95: surr ? surr.q95 : null,
      };
    }).filter((r): r is PredictionRow => r !== null);
  }, [evaluate.data, predict.data]);

  const meta: PredictionPanelMeta | undefined = evaluate.data
    ? {
        thermal: evaluate.data.thermal,
        stall: evaluate.data.stall,
      }
    : undefined;

  const handlePredict = () => {
    // Plumb the (optional) δ_ops override through to both routes.
    // SCHEMA_VERSION v7_1: δ_ops is a true LHS feature, so the
    // surrogate keeps its calibrated PIs across the entire slider
    // range; both /evaluate and /predict honour the override.
    const opsDuty = opsDutyOverride ?? null;
    // Schema v9: forward the (optional) payload mission-requirement
    // overrides to both routes so the deterministic median and the
    // surrogate PI band share the same mass/power budget.
    const shared = {
      design,
      scenario_name: scenarioName,
      operational_duty_cycle: opsDuty,
      payload_mass_kg: payloadMassOverride,
      payload_power_w: payloadPowerOverride,
    };
    evaluate.mutate(shared);
    predict.mutate(shared);
  };

  const evaluatorOnlyMode = predict.data?.mode === "evaluator_only";

  const isPending = evaluate.isPending;
  const surrogatePending = !evaluate.isPending && predict.isPending;
  // Bubble up the most informative error: evaluator failure is fatal
  // (no chart at all); a surrogate-only failure still lets the chart
  // render with the median, just without the PI band.
  const error = evaluate.error ?? (rows === undefined ? predict.error : null);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Mission inputs</CardTitle>
          <CardDescription>
            Scenario, scientific payload, and drive duty cycle.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <MissionInputsPanel disabled={isPending} embedded />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Compare with real rovers</CardTitle>
          <CardDescription>
            Select registry rovers to overlay their evaluator output on the
            performance chart under the current scenario and mission inputs.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RegistryOverlayPicker showHeader={false} />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Design inputs</CardTitle>
            <CardDescription>
              Configure the candidate rover within the calibrated design space.
              Coloured marks on sliders show selected real-rover values.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <DesignForm disabled={isPending} ticks={formTicks} />
            <Button
              type="button"
              onClick={handlePredict}
              disabled={isPending}
              className="w-full"
              size="lg"
            >
              <Play className="mr-2 h-4 w-4" />
              {isPending ? "Evaluating…" : "Predict performance"}
            </Button>
          </CardContent>
        </Card>

        <PredictionPanel
          rows={rows}
          meta={meta}
          isPending={isPending}
          error={error}
          surrogatePending={surrogatePending}
          overlays={overlays}
          overlayLoading={overlayInputs.length > 0 && overlayQueries.isPending}
          banner={evaluatorOnlyMode ? <NoPiBanner /> : null}
        />
      </div>
    </div>
  );
}
