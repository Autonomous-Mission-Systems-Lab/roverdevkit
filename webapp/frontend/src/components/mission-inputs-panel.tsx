import { RotateCcw } from "lucide-react";

import { DesignSliderField } from "@/components/design-slider-field";
import { MissionDurationPanel } from "@/components/mission-duration-panel";
import { OperationsPanel } from "@/components/operations-panel";
import { ScenarioPicker } from "@/components/scenario-picker";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useScenarios } from "@/hooks/use-scenarios";
import { useDesignStore } from "@/store/design-store";
import { OBSTACLE_BOUNDS, PAYLOAD_BOUNDS } from "@/types/api";

interface MissionInputsPanelProps {
  disabled?: boolean;
  /** When true, omit the in-panel section header (page supplies Card title). */
  embedded?: boolean;
  /** When false, omit the helper line under the section title. */
  showDescription?: boolean;
}

/**
 * Mission Inputs: the requirements a mission *sets* (scenario, scientific
 * payload, operational duty cycle) rather than the design variables a
 * rover engineer *trades* (wheels, mass, power, …).
 */
export function MissionInputsPanel({
  disabled,
  embedded = false,
  showDescription = true,
}: MissionInputsPanelProps) {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const opsDutyOverride = useDesignStore((s) => s.opsDutyOverride);
  const payloadMassOverride = useDesignStore((s) => s.payloadMassOverride);
  const payloadPowerOverride = useDesignStore((s) => s.payloadPowerOverride);
  const missionDurationOverride = useDesignStore((s) => s.missionDurationOverride);
  const requiredObstacleOverride = useDesignStore((s) => s.requiredObstacleOverride);
  const setRequiredObstacleOverride = useDesignStore(
    (s) => s.setRequiredObstacleOverride,
  );
  const setPayloadMassOverride = useDesignStore((s) => s.setPayloadMassOverride);
  const setPayloadPowerOverride = useDesignStore(
    (s) => s.setPayloadPowerOverride,
  );
  const clearOpsDutyOverride = useDesignStore((s) => s.clearOpsDutyOverride);
  const clearPayloadOverrides = useDesignStore((s) => s.clearPayloadOverrides);
  const clearMissionDurationOverride = useDesignStore(
    (s) => s.clearMissionDurationOverride,
  );
  const clearRequiredObstacleOverride = useDesignStore(
    (s) => s.clearRequiredObstacleOverride,
  );

  const { data, isLoading } = useScenarios();
  const scenario = data?.scenarios.find(
    (s) => s.scenario.name === scenarioName,
  )?.scenario;
  const massDefault = scenario?.payload_mass_kg ?? 0;
  const powerDefault = scenario?.payload_power_w ?? 0;

  const obstacleDefault = scenario?.required_obstacle_height_m ?? 0;
  const massValue = payloadMassOverride ?? massDefault;
  const powerValue = payloadPowerOverride ?? powerDefault;
  const obstacleValue = requiredObstacleOverride ?? obstacleDefault;
  const hasOverrides =
    opsDutyOverride !== null ||
    payloadMassOverride !== null ||
    payloadPowerOverride !== null ||
    missionDurationOverride !== null ||
    requiredObstacleOverride !== null;

  const massBounds = PAYLOAD_BOUNDS.payload_mass_kg;
  const powerBounds = PAYLOAD_BOUNDS.payload_power_w;
  const obstacleBounds = OBSTACLE_BOUNDS.required_obstacle_height_m;

  const clearAllOverrides = () => {
    clearOpsDutyOverride();
    clearPayloadOverrides();
    clearMissionDurationOverride();
    clearRequiredObstacleOverride();
  };

  return (
    <div className="space-y-5">
      {!embedded ? (
        <div>
          <Label className="text-sm font-semibold">Mission inputs</Label>
          {showDescription ? (
            <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
              Scenario, duration, payload, and drive duty cycle.
            </p>
          ) : null}
        </div>
      ) : null}

      <ScenarioPicker />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <MissionDurationPanel disabled={disabled} />
        <OperationsPanel disabled={disabled} />

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

        <DesignSliderField
          id="required_obstacle_height_m"
          label={obstacleBounds.label}
          unit={obstacleBounds.unit}
          description={obstacleBounds.description}
          min={obstacleBounds.min}
          max={obstacleBounds.max}
          step={obstacleBounds.step}
          value={obstacleValue}
          disabled={disabled || isLoading}
          onChange={(v) => setRequiredObstacleOverride(v)}
        />
      </div>

      {hasOverrides ? (
        <div className="flex justify-end">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={clearAllOverrides}
          >
            <RotateCcw className="mr-1 h-3 w-3" /> Reset to scenario defaults
          </Button>
        </div>
      ) : null}
    </div>
  );
}
