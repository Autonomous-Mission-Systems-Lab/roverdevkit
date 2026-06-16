import { DesignSliderField } from "@/components/design-slider-field";
import { useScenarios } from "@/hooks/use-scenarios";
import { useDesignStore } from "@/store/design-store";

/**
 * Per-scenario operational duty cycle override (δ_ops).
 */
export function OperationsPanel({ disabled }: { disabled?: boolean }) {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const setOpsDutyOverride = useDesignStore((s) => s.setOpsDutyOverride);

  const { data, isLoading } = useScenarios();
  const scenario = data?.scenarios.find(
    (s) => s.scenario.name === scenarioName,
  )?.scenario;
  const scenarioDefault = scenario?.operational_duty_cycle ?? 0.15;

  const value = opsDutyOverride ?? scenarioDefault;

  return (
    <DesignSliderField
      id="operational_duty_cycle"
      label="Operational duty cycle"
      unit=""
      description={`δ_ops, drive duty cycle. Scenario default: ${scenarioDefault.toFixed(2)}.`}
      min={0.0}
      max={0.6}
      step={0.01}
      value={value}
      disabled={disabled || isLoading}
      onChange={(v) => setOpsDutyOverride(v)}
    />
  );
}
