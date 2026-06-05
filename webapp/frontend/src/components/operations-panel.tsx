import { RotateCcw } from "lucide-react";

import { DesignSliderField } from "@/components/design-slider-field";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useScenarios } from "@/hooks/use-scenarios";
import { useDesignStore } from "@/store/design-store";

/**
 * Per-scenario operations override.
 *
 * Schema v6 (v6 schema update) introduced a v5 ``drive_duty_cycle`` -> two-
 * field split (designed vs. operational duty). Schema v7 (v6 schema update
 * follow-up) collapsed that split back into a single per-scenario
 * ``operational_duty_cycle`` after ``designed_duty_cycle`` turned out
 * to do no engineering work in the v6 mass model. Drive duty cycle
 * is now the *only* duty knob, and it lives entirely on the
 * scenario.
 *
 * The four canonical scenarios ship calibrated δ_ops values (mare
 * 0.30, crater 0.20, highland 0.15, polar 0.05); this panel lets a
 * user override δ_ops for a single `/evaluate` + `/predict`
 * round-trip without editing the scenario YAML.
 *
 * SCHEMA_VERSION v7_1 (v7_1 schema follow-on): δ_ops is now an LHS
 * feature in the surrogate dataset (uniform [0, 0.6] independent of
 * scenario family), so the override stays on the surrogate path and
 * keeps its calibrated PIs across the entire slider range. Pre-v7_1
 * any off-default override forced an evaluator-only fallback.
 */
export function OperationsPanel({ disabled }: { disabled?: boolean }) {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const setOpsDutyOverride = useDesignStore((s) => s.setOpsDutyOverride);
  const clearOpsDutyOverride = useDesignStore((s) => s.clearOpsDutyOverride);

  const { data, isLoading } = useScenarios();
  const scenario = data?.scenarios.find(
    (s) => s.scenario.name === scenarioName,
  )?.scenario;
  const scenarioDefault = scenario?.operational_duty_cycle ?? 0.15;

  const value = opsDutyOverride ?? scenarioDefault;
  const overridden = opsDutyOverride !== null;

  return (
    <div className="space-y-3 rounded-md border border-[var(--color-border)] p-3">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-sm font-medium">Operations (per query)</Label>
        {overridden ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={() => clearOpsDutyOverride()}
            className="h-7 px-2 text-xs"
          >
            <RotateCcw className="mr-1 h-3 w-3" /> Reset to scenario default
          </Button>
        ) : (
          <span className="text-[0.65rem] text-[var(--color-muted-foreground)]">
            using scenario default ({scenarioDefault.toFixed(2)})
          </span>
        )}
      </div>

      <DesignSliderField
        id="operational_duty_cycle"
        label="Operational duty cycle"
        unit=""
        description="δ_ops, drive duty cycle."
        min={0.0}
        max={0.6}
        step={0.01}
        value={value}
        disabled={disabled || isLoading}
        onChange={(v) => setOpsDutyOverride(v)}
      />

      <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
        δ_eff ={" "}
        <span className="font-medium tabular-nums text-[var(--color-foreground)]">
          {value.toFixed(2)}
        </span>
      </p>
    </div>
  );
}
