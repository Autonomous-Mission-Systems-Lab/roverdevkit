import { DesignSliderField } from "@/components/design-slider-field";
import { useScenarios } from "@/hooks/use-scenarios";
import {
  formatLunationContext,
  LUNAR_SYNODIC_DAYS,
  MISSION_DURATION_BOUNDS,
  SUNLIT_HALF_DAYS,
} from "@/lib/lunar";
import { useDesignStore } from "@/store/design-store";
import { cn } from "@/lib/utils";

interface MissionDurationPanelProps {
  disabled?: boolean;
}

/**
 * Per-query mission-duration override with lunar-day context.
 */
export function MissionDurationPanel({ disabled }: MissionDurationPanelProps) {
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const durationOverride = useDesignStore((s) => s.missionDurationOverride);
  const setMissionDurationOverride = useDesignStore(
    (s) => s.setMissionDurationOverride,
  );

  const { data, isLoading } = useScenarios();
  const scenario = data?.scenarios.find(
    (s) => s.scenario.name === scenarioName,
  )?.scenario;
  const scenarioDefault = scenario?.mission_duration_earth_days ?? 14;

  const value = durationOverride ?? scenarioDefault;

  return (
    <div className="space-y-1.5">
      <DesignSliderField
        id="mission_duration_earth_days"
        label="Mission duration"
        unit="d"
        description={`Simulation window. Scenario default: ${scenarioDefault.toFixed(0)} d.`}
        min={MISSION_DURATION_BOUNDS.min}
        max={MISSION_DURATION_BOUNDS.max}
        step={MISSION_DURATION_BOUNDS.step}
        value={value}
        disabled={disabled || isLoading}
        onChange={(v) => setMissionDurationOverride(v)}
      />

      <div className="flex flex-wrap items-center gap-1.5">
        <PresetChip
          label="Sunlit half"
          active={Math.abs(value - SUNLIT_HALF_DAYS) < 0.25}
          disabled={disabled || isLoading}
          onClick={() => setMissionDurationOverride(SUNLIT_HALF_DAYS)}
        />
        <PresetChip
          label="Full lunation"
          active={Math.abs(value - LUNAR_SYNODIC_DAYS) < 0.25}
          disabled={disabled || isLoading}
          onClick={() => setMissionDurationOverride(LUNAR_SYNODIC_DAYS)}
        />
        <span className="text-[0.65rem] text-[var(--color-muted-foreground)]">
          {formatLunationContext(value)}
        </span>
      </div>
    </div>
  );
}

function PresetChip({
  label,
  active,
  disabled,
  onClick,
}: {
  label: string;
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "rounded-md border px-2 py-0.5 text-[0.65rem] transition-colors",
        active
          ? "border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-foreground)]"
          : "border-[var(--color-border)] text-[var(--color-muted-foreground)] hover:bg-[var(--color-muted)]/40",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      {label}
    </button>
  );
}
