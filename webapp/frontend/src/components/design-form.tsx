import {
  DesignSliderField,
  type SliderTick,
} from "@/components/design-slider-field";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useDesignStore } from "@/store/design-store";
import { DESIGN_BOUNDS, type DesignVector } from "@/types/api";

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
 * Twelve-field design vector form.
 *
 * Each continuous field is a slider paired with an editable numeric
 * input (the slider scrubs, the input is the precision escape
 * hatch). The discrete `n_wheels` choice is rendered as a segmented
 * control (4 / 6) since a 2-step slider is awkward and hides the
 * fact that there are only two valid values.
 *
 * When real-rover overlays are enabled in the right-hand chart,
 * coloured tick marks appear above each slider track at the selected
 * rovers' values, using the same colours as the chart's overlay
 * markers so the cross-component association is preserved.
 */
export function DesignForm({ disabled, ticks = {} }: DesignFormProps) {
  const design = useDesignStore((s) => s.design);
  const setDesignField = useDesignStore((s) => s.setDesignField);
  const resetDesign = useDesignStore((s) => s.resetDesign);

  const continuousFields = (
    Object.keys(DESIGN_BOUNDS) as (keyof DesignVector)[]
  ).filter((k) => k !== "n_wheels");

  return (
    <div className="space-y-5">
      <WheelCountField
        value={design.n_wheels}
        ticks={ticks.n_wheels ?? []}
        disabled={disabled}
        onChange={(v) => setDesignField("n_wheels", v)}
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

interface WheelCountFieldProps {
  value: 4 | 6;
  ticks: SliderTick[];
  disabled?: boolean;
  onChange: (v: 4 | 6) => void;
}

/** Segmented control for the discrete-valued `n_wheels` field. */
function WheelCountField({
  value,
  ticks,
  disabled,
  onChange,
}: WheelCountFieldProps) {
  const options: Array<{ value: 4 | 6; label: string }> = [
    { value: 4, label: "4 wheels" },
    { value: 6, label: "6 wheels" },
  ];
  // Group ticks by which option they land on; we tag each option
  // with a small dot per rover whose design uses that wheel count.
  const ticksByValue = new Map<4 | 6, SliderTick[]>();
  for (const tick of ticks) {
    const v = tick.value === 4 ? 4 : 6;
    const existing = ticksByValue.get(v) ?? [];
    existing.push(tick);
    ticksByValue.set(v, existing);
  }

  return (
    <div className="space-y-1.5">
      <Label className="flex items-center justify-between">
        <span>Wheel count</span>
        <span className="text-xs text-[var(--color-muted-foreground)]">—</span>
      </Label>
      <div
        role="radiogroup"
        aria-label="Wheel count"
        className="inline-flex rounded-md border bg-[var(--color-muted)]/30 p-0.5"
      >
        {options.map((opt) => {
          const active = value === opt.value;
          const optTicks = ticksByValue.get(opt.value) ?? [];
          return (
            <button
              key={opt.value}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(opt.value)}
              className={cn(
                "relative flex items-center gap-2 rounded px-3 py-1.5 text-sm transition-colors",
                active
                  ? "bg-[var(--color-background)] font-medium shadow-sm"
                  : "text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]",
                disabled && "cursor-not-allowed opacity-50",
              )}
            >
              <span>{opt.label}</span>
              {optTicks.length > 0 ? (
                <span className="flex items-center gap-1">
                  {optTicks.map((tick) => (
                    <span
                      key={tick.rover_name}
                      title={`${tick.rover_name}`}
                      aria-hidden
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: tick.color }}
                    />
                  ))}
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
      <p className="text-[0.65rem] text-[var(--color-muted-foreground)]">
        N_w, mobility wheel count.
      </p>
    </div>
  );
}
