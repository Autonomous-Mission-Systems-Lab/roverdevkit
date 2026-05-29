import { RotateCcw } from "lucide-react";

import { DesignSliderField } from "@/components/design-slider-field";
import { OperationsPanel } from "@/components/operations-panel";
import { ScenarioPicker } from "@/components/scenario-picker";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useScenarios } from "@/hooks/use-scenarios";
import { useDesignStore } from "@/store/design-store";
import { PAYLOAD_BOUNDS } from "@/types/api";

/**
 * Mission Inputs: the requirements a mission *sets* (scenario, scientific
 * payload, operational duty cycle) rather than the design variables a
 * rover engineer *trades* (wheels, mass, power, …).
 *
 * Schema v9 promoted scientific payload from a per-rover `chassis_mass_kg`
 * convention to two explicit mission requirements carried on
 * `MissionScenario` (`payload_mass_kg` / `payload_power_w`). Each canonical
 * scenario ships a class-typical default; this panel lets the user override
 * both for a single round-trip without editing the scenario catalogue.
 * Payload mass is added to total vehicle mass as a top-level line item
 * outside the dry-mass growth margin; payload power adds to the continuous
 * electrical load. Both are LHS-sampled surrogate inputs over [0, 30], so
 * any in-bounds override stays on the surrogate path with calibrated PIs.
 */
export function MissionInputsPanel({ disabled }: { disabled?: boolean }) {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const payloadMassOverride = useDesignStore((s) => s.payloadMassOverride);
  const payloadPowerOverride = useDesignStore((s) => s.payloadPowerOverride);
  const setPayloadMassOverride = useDesignStore((s) => s.setPayloadMassOverride);
  const setPayloadPowerOverride = useDesignStore(
    (s) => s.setPayloadPowerOverride,
  );
  const clearPayloadOverrides = useDesignStore((s) => s.clearPayloadOverrides);

  const { data, isLoading } = useScenarios();
  const scenario = data?.scenarios.find(
    (s) => s.scenario.name === scenarioName,
  )?.scenario;
  const massDefault = scenario?.payload_mass_kg ?? 0;
  const powerDefault = scenario?.payload_power_w ?? 0;

  const massValue = payloadMassOverride ?? massDefault;
  const powerValue = payloadPowerOverride ?? powerDefault;
  const overridden =
    payloadMassOverride !== null || payloadPowerOverride !== null;

  const massBounds = PAYLOAD_BOUNDS.payload_mass_kg;
  const powerBounds = PAYLOAD_BOUNDS.payload_power_w;

  return (
    <div className="space-y-4">
      <div>
        <Label className="text-sm font-semibold">Mission inputs</Label>
        <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
          Requirements set by the mission (scenario, scientific payload,
          operational duty) — not design variables the rover engineer trades.
        </p>
      </div>

      <ScenarioPicker />

      <div className="space-y-3 rounded-md border border-[var(--color-border)] p-3">
        <div className="flex items-center justify-between gap-2">
          <Label className="text-sm font-medium">Scientific payload</Label>
          {overridden ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={disabled}
              onClick={() => clearPayloadOverrides()}
              className="h-7 px-2 text-xs"
            >
              <RotateCcw className="mr-1 h-3 w-3" /> Reset to scenario default
            </Button>
          ) : (
            <span className="text-[0.65rem] text-[var(--color-muted-foreground)]">
              using scenario defaults ({massDefault.toFixed(1)} kg ·{" "}
              {powerDefault.toFixed(1)} W)
            </span>
          )}
        </div>

        <DesignSliderField
          id="payload_mass_kg"
          label={massBounds.label}
          unit={massBounds.unit}
          description={massBounds.description}
          min={massBounds.min}
          max={massBounds.max}
          step={massBounds.step}
          value={massValue}
          disabled={disabled || isLoading}
          onChange={(v) => setPayloadMassOverride(v)}
        />

        <DesignSliderField
          id="payload_power_w"
          label={powerBounds.label}
          unit={powerBounds.unit}
          description={powerBounds.description}
          min={powerBounds.min}
          max={powerBounds.max}
          step={powerBounds.step}
          value={powerValue}
          disabled={disabled || isLoading}
          onChange={(v) => setPayloadPowerOverride(v)}
        />
      </div>

      <OperationsPanel disabled={disabled} />
    </div>
  );
}
