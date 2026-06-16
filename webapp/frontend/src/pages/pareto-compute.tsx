import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Square } from "lucide-react";

import { MissionInputsPanel } from "@/components/mission-inputs-panel";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  useCancelOptimize,
  useOptimizeResult,
  useStartOptimize,
} from "@/hooks/use-optimize";
import { apiUrl } from "@/lib/api";
import { ParetoExplorer } from "@/pages/pareto-explorer";
import { useDesignStore } from "@/store/design-store";
import { useParetoStore } from "@/store/pareto-store";
import type {
  ConstraintSense,
  ObjectiveDirection,
  OptimizeCheckpointOut,
  OptimizeJobResponse,
  OptimizeResultResponse,
  PrimaryTarget,
} from "@/types/api";
import { PRIMARY_REGRESSION_TARGET_ORDER, TARGET_META } from "@/types/api";

/**
 * Hard ceiling enforced by the backend optimize route's evaluator cap
 * (see `webapp/backend/routes/optimize.py`). Lives here so the UI can
 * warn the user before they queue a job that will 422.
 */
const EVALUATOR_EVAL_CAP = 5000;

const DEFAULT_OBJECTIVES: Record<PrimaryTarget, ObjectiveDirection> = {
  range_km: "max",
  energy_margin_raw_pct: "max",
  slope_capability_deg: "max",
  total_mass_kg: "min",
};

type ObjectiveState = Record<PrimaryTarget, boolean>;

interface ConstraintState {
  enabled: boolean;
  sense: ConstraintSense;
  value: number;
}

// The 0.1 km range floor is enabled by default so NSGA-II never
// rewards stalled designs (``range_km = 0``) just because they happen
// to score well on slope capability. Disabling the constraint puts
// stalled designs back on the Pareto front when they win another
// objective, which is usually not what a user wants. The other
// constraints are off by default and intended as optional add-ons.
const DEFAULT_CONSTRAINTS: Record<PrimaryTarget, ConstraintState> = {
  range_km: { enabled: true, sense: "min", value: 0.1 },
  energy_margin_raw_pct: { enabled: false, sense: "min", value: 0.0 },
  slope_capability_deg: { enabled: false, sense: "min", value: 10.0 },
  total_mass_kg: { enabled: false, sense: "max", value: 40.0 },
};

export function ParetoCompute() {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const payloadMassOverride = useDesignStore((s) => s.payloadMassOverride);
  const payloadPowerOverride = useDesignStore((s) => s.payloadPowerOverride);
  const missionDurationOverride = useDesignStore((s) => s.missionDurationOverride);

  const [populationSize, setPopulationSize] = useState(32);
  const [nGenerations, setNGenerations] = useState(50);
  const [seed, setSeed] = useState(0);
  const [objectiveEnabled, setObjectiveEnabled] = useState<ObjectiveState>({
    range_km: true,
    energy_margin_raw_pct: false,
    slope_capability_deg: true,
    total_mass_kg: true,
  });
  const [objectiveDirections, setObjectiveDirections] =
    useState<Record<PrimaryTarget, ObjectiveDirection>>(DEFAULT_OBJECTIVES);
  const [constraints, setConstraints] =
    useState<Record<PrimaryTarget, ConstraintState>>(DEFAULT_CONSTRAINTS);

  const [job, setJob] = useState<OptimizeJobResponse | null>(null);
  const [checkpoints, setCheckpoints] = useState<OptimizeCheckpointOut[]>([]);
  const [result, setResult] = useState<OptimizeResultResponse | null>(null);
  const [progressOpen, setProgressOpen] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const startOptimize = useStartOptimize();
  const fetchResult = useOptimizeResult();
  const cancelOptimize = useCancelOptimize();
  const setFrontFromResult = useParetoStore((s) => s.setFromOptimizeResult);

  useEffect(
    () => () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    },
    [],
  );

  const selectedObjectives = useMemo(
    () =>
      PRIMARY_REGRESSION_TARGET_ORDER.filter((target) => objectiveEnabled[target]),
    [objectiveEnabled],
  );
  const evaluationBudget = populationSize * nGenerations;
  const latestCheckpoint = checkpoints.at(-1);
  const terminalStatus =
    result?.status ??
    (streamError ? "failed" : job?.status === "queued" ? "running" : null);

  const handleRun = async () => {
    if (selectedObjectives.length === 0) {
      setStreamError("Pick at least one objective.");
      return;
    }
    closeStream();
    setCheckpoints([]);
    setResult(null);
    setStreamError(null);
    setProgressOpen(true);
    const objectivePayload = selectedObjectives.map((target) => ({
      target,
      direction: objectiveDirections[target],
    }));

    try {
      const queued = await startOptimize.mutateAsync({
        scenario_name: scenarioName,
        backend: "evaluator",
        objectives: objectivePayload,
        constraints: PRIMARY_REGRESSION_TARGET_ORDER.filter(
          (target) => constraints[target].enabled,
        ).map((target) => ({
          target,
          sense: constraints[target].sense,
          value: constraints[target].value,
        })),
        population_size: populationSize,
        n_generations: nGenerations,
        seed,
        operational_duty_cycle: opsDutyOverride ?? null,
        // Schema v9: every NSGA-II candidate is scored carrying this
        // payload mass/power, so the front reflects the mission's real
        // mass budget instead of floating chassis to the LHS floor.
        payload_mass_kg: payloadMassOverride,
        payload_power_w: payloadPowerOverride,
        mission_duration_earth_days: missionDurationOverride,
      });
      setJob(queued);
      openStream(queued, objectivePayload);
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "Failed to start job.");
    }
  };

  const openStream = (
    queued: OptimizeJobResponse,
    objectivePayload: Array<{
      target: PrimaryTarget;
      direction: ObjectiveDirection;
    }>,
  ) => {
    const source = new EventSource(apiUrl(queued.stream_url));
    eventSourceRef.current = source;
    source.addEventListener("checkpoint", (event) => {
      const checkpoint = JSON.parse(
        (event as MessageEvent<string>).data,
      ) as OptimizeCheckpointOut;
      setCheckpoints((prev) => [...prev, checkpoint]);
    });
    for (const status of ["completed", "cancelled", "failed"] as const) {
      source.addEventListener(status, async () => {
        closeStream();
        try {
          const finalResult = await fetchResult.mutateAsync(queued.result_url);
          setResult(finalResult);
          if (finalResult.status === "completed") {
            setFrontFromResult(finalResult, scenarioName, objectivePayload);
          }
        } catch (err) {
          setStreamError(
            err instanceof Error ? err.message : "Failed to fetch optimization result.",
          );
        }
      });
    }
    source.onerror = () => {
      closeStream();
      setStreamError("Optimization stream disconnected.");
    };
  };

  const handleCancel = async () => {
    if (!job) return;
    try {
      await cancelOptimize.mutateAsync(job.cancel_url);
    } catch (err) {
      setStreamError(err instanceof Error ? err.message : "Failed to cancel job.");
    }
  };

  const closeStream = () => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  };

  return (
    <>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(20rem,0.85fr)_minmax(0,1.45fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Find optimized designs</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <MissionInputsPanel
                disabled={startOptimize.isPending}
                showDescription={false}
              />
              <ObjectiveEditor
                enabled={objectiveEnabled}
                directions={objectiveDirections}
                setEnabled={setObjectiveEnabled}
                setDirections={setObjectiveDirections}
              />
              <ConstraintEditor constraints={constraints} setConstraints={setConstraints} />
              <BudgetEditor
                populationSize={populationSize}
                nGenerations={nGenerations}
                seed={seed}
                setPopulationSize={setPopulationSize}
                setNGenerations={setNGenerations}
                setSeed={setSeed}
              />
              {evaluationBudget > EVALUATOR_EVAL_CAP ? (
                <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
                  Live NSGA-II is capped at {EVALUATOR_EVAL_CAP.toLocaleString()}{" "}
                  evaluator calls (~2 min wall clock). Lower population or
                  generations before running, or run{" "}
                  <code>make pareto-fronts</code> offline for higher budgets.
                </p>
              ) : null}
              <Button
                type="button"
                onClick={handleRun}
                disabled={
                  startOptimize.isPending ||
                  selectedObjectives.length === 0 ||
                  evaluationBudget > EVALUATOR_EVAL_CAP
                }
                className="w-full"
                size="lg"
              >
                <Play className="mr-2 h-4 w-4" />
                {startOptimize.isPending ? "Queueing job..." : "Run NSGA-II"}
              </Button>
              {streamError ? (
                <p className="text-sm text-[var(--color-destructive)]">{streamError}</p>
              ) : null}
            </CardContent>
          </Card>
        </div>

        <ParetoExplorer />
      </div>

      <Dialog open={progressOpen} onOpenChange={setProgressOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>NSGA-II progress</DialogTitle>
            <DialogDescription>
              Streaming per-generation checkpoints from the backend job.
            </DialogDescription>
          </DialogHeader>
          {latestCheckpoint ? (
            <CheckpointSummary checkpoint={latestCheckpoint} />
          ) : (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Waiting for the first generation...
            </p>
          )}
          {terminalStatus ? (
            <p className="text-sm">
              Job status: <span className="font-medium">{terminalStatus}</span>
            </p>
          ) : null}
          {streamError ? (
            <p className="text-sm text-[var(--color-destructive)]">{streamError}</p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setProgressOpen(false)}
            >
              Hide
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={handleCancel}
              disabled={!job || Boolean(result) || cancelOptimize.isPending}
            >
              <Square className="mr-2 h-4 w-4" />
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ObjectiveEditor({
  enabled,
  directions,
  setEnabled,
  setDirections,
}: {
  enabled: ObjectiveState;
  directions: Record<PrimaryTarget, ObjectiveDirection>;
  setEnabled: (next: ObjectiveState) => void;
  setDirections: (next: Record<PrimaryTarget, ObjectiveDirection>) => void;
}) {
  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-medium">Objectives</h3>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Pick one or more metrics and whether NSGA-II should minimize or maximize them.
        </p>
      </div>
      <div className="space-y-2">
        {PRIMARY_REGRESSION_TARGET_ORDER.map((target) => (
          <div
            key={target}
            className="grid grid-cols-[1fr_8rem] items-center gap-3 rounded-md border p-3"
          >
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                checked={enabled[target]}
                onChange={(event) =>
                  setEnabled({ ...enabled, [target]: event.target.checked })
                }
                className="mt-1"
              />
              <span>
                <span className="font-medium">{TARGET_META[target].label}</span>
                <span className="block text-xs text-[var(--color-muted-foreground)]">
                  {TARGET_META[target].description}
                </span>
              </span>
            </label>
            <Select
              value={directions[target]}
              onValueChange={(v) =>
                setDirections({
                  ...directions,
                  [target]: v as ObjectiveDirection,
                })
              }
              disabled={!enabled[target]}
            >
              <SelectTrigger aria-label={`${TARGET_META[target].label} direction`}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="max">Maximize</SelectItem>
                <SelectItem value="min">Minimize</SelectItem>
              </SelectContent>
            </Select>
          </div>
        ))}
      </div>
    </section>
  );
}

function ConstraintEditor({
  constraints,
  setConstraints,
}: {
  constraints: Record<PrimaryTarget, ConstraintState>;
  setConstraints: (next: Record<PrimaryTarget, ConstraintState>) => void;
}) {
  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-medium">Constraints</h3>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Optional feasibility thresholds; enabled rows become NSGA-II constraints.
        </p>
      </div>
      <div className="space-y-2">
        {PRIMARY_REGRESSION_TARGET_ORDER.map((target) => {
          const row = constraints[target];
          return (
            <div
              key={target}
              className="grid grid-cols-[1fr_7rem_7rem] items-center gap-2 rounded-md border p-3"
            >
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={row.enabled}
                  onChange={(event) =>
                    setConstraints({
                      ...constraints,
                      [target]: { ...row, enabled: event.target.checked },
                    })
                  }
                />
                {TARGET_META[target].label}
              </label>
              <Select
                value={row.sense}
                onValueChange={(v) =>
                  setConstraints({
                    ...constraints,
                    [target]: { ...row, sense: v as ConstraintSense },
                  })
                }
                disabled={!row.enabled}
              >
                <SelectTrigger aria-label={`${TARGET_META[target].label} constraint`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="min">At least</SelectItem>
                  <SelectItem value="max">At most</SelectItem>
                </SelectContent>
              </Select>
              <Input
                type="number"
                value={row.value}
                onChange={(event) =>
                  setConstraints({
                    ...constraints,
                    [target]: {
                      ...row,
                      value: Number(event.target.value),
                    },
                  })
                }
                disabled={!row.enabled}
                aria-label={`${TARGET_META[target].label} threshold`}
              />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function BudgetEditor({
  populationSize,
  nGenerations,
  seed,
  setPopulationSize,
  setNGenerations,
  setSeed,
}: {
  populationSize: number;
  nGenerations: number;
  seed: number;
  setPopulationSize: (value: number) => void;
  setNGenerations: (value: number) => void;
  setSeed: (value: number) => void;
}) {
  return (
    <section className="space-y-4">
      <div>
        <h3 className="text-sm font-medium">NSGA-II budget</h3>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Evaluation budget: {(populationSize * nGenerations).toLocaleString()} fitness calls.
        </p>
      </div>
      <div className="space-y-2">
        <Label>Population size: {populationSize}</Label>
        <Slider
          min={4}
          max={300}
          step={4}
          value={[populationSize]}
          onValueChange={([value]) => setPopulationSize(value)}
        />
      </div>
      <div className="space-y-2">
        <Label>Generations: {nGenerations}</Label>
        <Slider
          min={1}
          max={500}
          step={1}
          value={[nGenerations]}
          onValueChange={([value]) => setNGenerations(value)}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="opt-seed">Random seed</Label>
        <Input
          id="opt-seed"
          type="number"
          min={0}
          value={seed}
          onChange={(event) => setSeed(Math.max(0, Number(event.target.value)))}
        />
      </div>
    </section>
  );
}

function CheckpointSummary({ checkpoint }: { checkpoint: OptimizeCheckpointOut }) {
  return (
    <div className="space-y-3 rounded-md border p-4">
      <div className="grid grid-cols-3 gap-3 text-sm">
        <Metric label="Generation" value={checkpoint.gen.toLocaleString()} />
        <Metric label="Front size" value={checkpoint.pareto_size.toLocaleString()} />
        <Metric label="Hypervolume" value={checkpoint.hypervolume.toPrecision(4)} />
      </div>
      <div className="space-y-1">
        <p className="text-xs font-medium text-[var(--color-muted-foreground)]">
          Best per objective
        </p>
        <div className="grid grid-cols-2 gap-2 text-sm">
          {Object.entries(checkpoint.best_per_objective).map(([target, value]) => (
            <Metric
              key={target}
              label={TARGET_META[target as PrimaryTarget]?.label ?? target}
              value={formatMetric(target as PrimaryTarget, value)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-[var(--color-muted-foreground)]">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  );
}

function formatMetric(target: PrimaryTarget, value: number): string {
  const unit = TARGET_META[target].unit;
  const digits = Math.abs(value) >= 100 ? 1 : 2;
  return `${value.toFixed(digits)} ${unit}`.trim();
}
