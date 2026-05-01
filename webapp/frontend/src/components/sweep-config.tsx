import { Plus, X } from "lucide-react";

import { ScenarioPicker } from "@/components/scenario-picker";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSweepStore } from "@/store/sweep-store";
import {
  DESIGN_BOUNDS,
  PRIMARY_REGRESSION_TARGET_ORDER,
  SWEEPABLE_VARIABLES,
  TARGET_META,
  type PrimaryTarget,
  type SweepBackend,
  type SweepableVariable,
} from "@/types/api";

const BACKEND_OPTIONS: Array<{
  value: SweepBackend;
  label: string;
  hint: string;
}> = [
  {
    value: "auto",
    label: "Auto",
    hint: "Evaluator below 200 cells, surrogate above.",
  },
  {
    value: "evaluator",
    label: "Evaluator (ground truth)",
    hint: "~40 ms per cell; capped at 2,500 cells.",
  },
  {
    value: "surrogate",
    label: "Surrogate (fast)",
    hint: "Vectorised quantile-XGBoost; capped at 40,000 cells.",
  },
];

/** Configuration panel for the parametric sweep page. */
export function SweepConfig({ disabled }: { disabled?: boolean }) {
  const target = useSweepStore((s) => s.target);
  const setTarget = useSweepStore((s) => s.setTarget);
  const xAxis = useSweepStore((s) => s.xAxis);
  const setXAxis = useSweepStore((s) => s.setXAxis);
  const setXVariable = useSweepStore((s) => s.setXVariable);
  const yAxis = useSweepStore((s) => s.yAxis);
  const setYAxis = useSweepStore((s) => s.setYAxis);
  const setYVariable = useSweepStore((s) => s.setYVariable);
  const setYEnabled = useSweepStore((s) => s.setYEnabled);
  const backend = useSweepStore((s) => s.backend);
  const setBackend = useSweepStore((s) => s.setBackend);

  const cellCount = xAxis.n_points * (yAxis ? yAxis.n_points : 1);

  return (
    <div className="space-y-6">
      <ScenarioPicker />

      <div className="space-y-2">
        <Label htmlFor="sweep-target">Performance metric</Label>
        <Select
          value={target}
          onValueChange={(v) => setTarget(v as PrimaryTarget)}
          disabled={disabled}
        >
          <SelectTrigger id="sweep-target" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PRIMARY_REGRESSION_TARGET_ORDER.map((t) => (
              <SelectItem key={t} value={t}>
                {TARGET_META[t].label} ({TARGET_META[t].unit})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          {TARGET_META[target].description}
        </p>
      </div>

      <AxisEditor
        title="X axis"
        axis={xAxis}
        onVariableChange={(v) => setXVariable(v)}
        onFieldChange={(patch) => setXAxis(patch)}
        disabled={disabled}
        excludedVariable={yAxis?.variable}
      />

      {yAxis ? (
        <div className="rounded-md border border-[var(--color-border)] p-3">
          <div className="mb-3 flex items-center justify-between">
            <Label className="text-sm font-medium">Y axis</Label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setYEnabled(false)}
              disabled={disabled}
              className="h-7 px-2 text-xs"
            >
              <X className="mr-1 h-3 w-3" /> Remove
            </Button>
          </div>
          <AxisEditor
            title=""
            axis={yAxis}
            onVariableChange={(v) => setYVariable(v)}
            onFieldChange={(patch) => setYAxis(patch)}
            disabled={disabled}
            excludedVariable={xAxis.variable}
          />
        </div>
      ) : (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setYEnabled(true)}
          disabled={disabled}
        >
          <Plus className="mr-1.5 h-3.5 w-3.5" />
          Add second axis (heatmap)
        </Button>
      )}

      <div className="space-y-2">
        <Label htmlFor="sweep-backend">Backend</Label>
        <Select
          value={backend}
          onValueChange={(v) => setBackend(v as SweepBackend)}
          disabled={disabled}
        >
          <SelectTrigger id="sweep-backend" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {BACKEND_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-[var(--color-muted-foreground)]">
          {BACKEND_OPTIONS.find((o) => o.value === backend)?.hint}
        </p>
      </div>

      <p className="text-xs text-[var(--color-muted-foreground)]">
        {cellCount.toLocaleString()} grid cells will be evaluated. The base
        design (every dimension not on an axis) is taken from your current
        single-design configuration.
      </p>
    </div>
  );
}

interface AxisEditorProps {
  title: string;
  axis: {
    variable: SweepableVariable;
    lo: number;
    hi: number;
    n_points: number;
  };
  onVariableChange: (v: SweepableVariable) => void;
  onFieldChange: (
    patch: Partial<{ lo: number; hi: number; n_points: number }>,
  ) => void;
  disabled?: boolean;
  excludedVariable?: SweepableVariable;
}

function AxisEditor({
  title,
  axis,
  onVariableChange,
  onFieldChange,
  disabled,
  excludedVariable,
}: AxisEditorProps) {
  const bounds = DESIGN_BOUNDS[axis.variable];

  return (
    <div className="space-y-3">
      {title ? <Label>{title}</Label> : null}

      <Select
        value={axis.variable}
        onValueChange={(v) => onVariableChange(v as SweepableVariable)}
        disabled={disabled}
      >
        <SelectTrigger className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SWEEPABLE_VARIABLES.map((v) => (
            <SelectItem key={v} value={v} disabled={v === excludedVariable}>
              {DESIGN_BOUNDS[v].label}
              {DESIGN_BOUNDS[v].unit ? ` (${DESIGN_BOUNDS[v].unit})` : ""}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <div className="grid grid-cols-3 gap-2">
        <div className="space-y-1">
          <Label className="text-xs text-[var(--color-muted-foreground)]">
            From
          </Label>
          <Input
            type="number"
            min={bounds.min}
            max={bounds.max}
            step={bounds.step}
            value={axis.lo}
            onChange={(e) => onFieldChange({ lo: Number(e.target.value) })}
            disabled={disabled}
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-[var(--color-muted-foreground)]">
            To
          </Label>
          <Input
            type="number"
            min={bounds.min}
            max={bounds.max}
            step={bounds.step}
            value={axis.hi}
            onChange={(e) => onFieldChange({ hi: Number(e.target.value) })}
            disabled={disabled}
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-[var(--color-muted-foreground)]">
            Points
          </Label>
          <Input
            type="number"
            min={2}
            max={200}
            step={1}
            value={axis.n_points}
            onChange={(e) =>
              onFieldChange({ n_points: Math.round(Number(e.target.value)) })
            }
            disabled={disabled}
          />
        </div>
      </div>
      <p className="text-xs text-[var(--color-muted-foreground)]">
        Schema range {bounds.min} – {bounds.max} {bounds.unit}.
      </p>
    </div>
  );
}
