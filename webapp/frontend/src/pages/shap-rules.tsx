import { useEffect, useState } from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useShapExplain } from "@/hooks/use-shap";
import { useDesignStore } from "@/store/design-store";
import type { PrimaryTarget, ShapFeatureScore } from "@/types/api";
import { PRIMARY_REGRESSION_TARGET_ORDER, TARGET_META } from "@/types/api";

export function ShapRules() {
  const design = useDesignStore((s) => s.design);
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const [target, setTarget] = useState<PrimaryTarget>("range_km");
  const {
    mutate: explainDesign,
    data: explanation,
    error,
    isError,
    isPending,
  } = useShapExplain();

  useEffect(() => {
    explainDesign({
      design,
      scenario_name: scenarioName,
      target,
      operational_duty_cycle: opsDutyOverride ?? null,
    });
  }, [design, explainDesign, opsDutyOverride, scenarioName, target]);

  return (
    <div className="mx-auto max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>Explain current design</CardTitle>
          <CardDescription>
            Uses SHAP-style feature attributions from the surrogate model to show
            which inputs push the selected prediction up or down for the active
            design and scenario from the Current design tab.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Select value={target} onValueChange={(v) => setTarget(v as PrimaryTarget)}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PRIMARY_REGRESSION_TARGET_ORDER.map((t) => (
                <SelectItem key={t} value={t}>
                  {TARGET_META[t].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {isPending ? (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Explaining current design...
            </p>
          ) : null}
          {isError ? (
            <p className="text-sm text-[var(--color-destructive)]">
              {error instanceof Error
                ? error.message
                : "Failed to explain current design."}
            </p>
          ) : explanation ? (
            <>
              <div className="grid grid-cols-2 gap-4 rounded-md border p-3 text-sm">
                <Metric
                  label="Prediction"
                  value={formatTarget(explanation.target, explanation.prediction)}
                />
                <Metric
                  label="Base value"
                  value={formatTarget(explanation.target, explanation.base_value)}
                />
              </div>
              <FeatureBars
                title="Largest feature contributions"
                rows={explanation.contributions}
              />
            </>
          ) : (
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Select a target to explain the current design.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function FeatureBars({
  title,
  rows,
}: {
  title: string;
  rows: ShapFeatureScore[];
}) {
  const scale = Math.max(...rows.map((row) => Math.abs(row.value)), 1e-12);
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium">{title}</h3>
      <div className="space-y-2">
        {rows.map((row) => {
          const width = `${Math.max(3, (Math.abs(row.value) / scale) * 100)}%`;
          const positive = row.value >= 0;
          return (
            <div key={row.feature} className="space-y-1">
              <div className="flex justify-between gap-3 text-xs">
                <span className="truncate">{prettifyFeature(row.feature)}</span>
                <span className="font-mono">{row.value.toPrecision(3)}</span>
              </div>
              <div className="h-2 rounded-full bg-[var(--color-muted)]">
                <div
                  className={
                    "h-2 rounded-full " +
                    (positive ? "bg-blue-600" : "bg-amber-600")
                  }
                  style={{ width }}
                />
              </div>
            </div>
          );
        })}
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

function formatTarget(target: PrimaryTarget, value: number): string {
  const unit = TARGET_META[target].unit;
  const digits = Math.abs(value) >= 100 ? 1 : 2;
  return `${value.toFixed(digits)} ${unit}`.trim();
}

function prettifyFeature(feature: string): string {
  return feature
    .replace(/^design_/, "")
    .replace(/^scenario_/, "")
    .replace(/_/g, " ");
}
