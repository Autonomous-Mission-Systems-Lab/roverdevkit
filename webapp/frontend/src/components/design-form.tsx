import {
  DesignSliderField,
  type SliderTick,
} from "@/components/design-slider-field";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useDesignStore } from "@/store/design-store";
import { DESIGN_BOUNDS, type DesignVector, type MobilityArchitecture } from "@/types/api";

/**
 * Per-field tick data for the registry-rover overlay. Keys are the
 * design-vector field names; each entry is the list of selected
 * rovers that have a numeric value to plot at that field.
 */
export type DesignFormTicks = Partial<Record<keyof DesignVector, SliderTick[]>>;

export interface DesignFormProps {
  disabled?: boolean;
  ticks?: DesignFormTicks;
}

/**
 * Design vector form with an explicit mobility-architecture selector.
 */
export function DesignForm({ disabled, ticks = {} }: DesignFormProps) {
  const design = useDesignStore((s) => s.design);
  const setDesignField = useDesignStore((s) => s.setDesignField);
  const resetDesign = useDesignStore((s) => s.resetDesign);

  const continuousFields = (
    Object.keys(DESIGN_BOUNDS) as (keyof typeof DESIGN_BOUNDS)[]
  ).filter((k) => k !== "n_wheels");

  return (
    <div className="space-y-5">
      <MobilityArchitectureField
        value={design.mobility_architecture}
        ticks={ticks.mobility_architecture ?? ticks.n_wheels ?? []}
        disabled={disabled}
        onChange={(v) => setDesignField("mobility_architecture", v)}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {continuousFields.map((key) => {
          const bounds = DESIGN_BOUNDS[key];
          const value = design[key] as number;
          const isInteger = key === "grouser_count";
          return (
            <DesignSliderField
              key={key}
              id={key}
              label={bounds.label}
              unit={bounds.unit}
              description={bounds.description}
              min={bounds.min}
              max={bounds.max}
              step={bounds.step}
              value={value}
              ticks={ticks[key] ?? []}
              disabled={disabled}
              format={isInteger ? (v) => `${Math.round(v)}` : undefined}
              sanitize={isInteger ? (v) => Math.round(v) : undefined}
              onChange={(v) => {
                if (isInteger) {
                  setDesignField("grouser_count", Math.round(v));
                } else {
                  (setDesignField as (k: typeof key, v: number) => void)(
                    key,
                    v,
                  );
                }
              }}
            />
          );
        })}
      </div>

      <div className="flex justify-end">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={disabled}
          onClick={resetDesign}
        >
          Reset to defaults
        </Button>
      </div>
    </div>
  );
}

interface MobilityArchitectureFieldProps {
  value: MobilityArchitecture;
  ticks: SliderTick[];
  disabled?: boolean;
  onChange: (v: MobilityArchitecture) => void;
}

function MobilityArchitectureField({
  value,
  ticks,
  disabled,
  onChange,
}: MobilityArchitectureFieldProps) {
  const options: Array<{ value: MobilityArchitecture; label: string }> = [
    { value: "rigid_4wheel", label: "Rigid 4-wheel" },
    { value: "rocker_bogie_6wheel", label: "Rocker-bogie 6-wheel" },
  ];

  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">Mobility architecture</Label>
      <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
        Sets wheel count, obstacle capability, and suspension mass proxy.
      </p>
      <div className="inline-flex rounded-md border p-0.5">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            disabled={disabled}
            className={cn(
              "rounded px-3 py-1.5 text-sm transition-colors",
              value === opt.value
                ? "bg-[var(--color-primary)] text-[var(--color-primary-foreground)]"
                : "text-[var(--color-muted-foreground)] hover:bg-[var(--color-muted)]/50",
            )}
            onClick={() => onChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>
      {ticks.length > 0 ? (
        <div className="flex flex-wrap gap-1 pt-1">
          {ticks.map((tick) => (
            <span
              key={tick.rover_name}
              className="inline-flex items-center gap-1 text-[0.65rem] text-[var(--color-muted-foreground)]"
            >
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: tick.color }}
              />
              {tick.rover_name}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
