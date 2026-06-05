import { Check } from "lucide-react";

import { Label } from "@/components/ui/label";
import { useRegistry } from "@/hooks/use-registry";
import { roverColor } from "@/lib/rover-colors";
import { cn } from "@/lib/utils";
import { useDesignStore } from "@/store/design-store";

interface RegistryOverlayPickerProps {
  /** Extra classes on the outer wrapper. */
  className?: string;
  /**
   * When true, wrap the control in a bordered panel (used inside
   * other cards). When false, render inline.
   */
  framed?: boolean;
  /** When false, omit the title and helper text (parent card supplies them). */
  showHeader?: boolean;
}

/**
 * Multi-select pill row for the registry overlay.
 *
 * Each pill is a real-rover entry from `/registry`; clicking one
 * toggles whether its prediction (run under the *user's* currently
 * selected scenario) is overlaid on the chart.
 */
export function RegistryOverlayPicker({
  className,
  framed = false,
  showHeader = true,
}: RegistryOverlayPickerProps) {
  const { data, isPending, isError } = useRegistry();
  const overlayRovers = useDesignStore((s) => s.overlayRovers);
  const toggleOverlayRover = useDesignStore((s) => s.toggleOverlayRover);

  const body = (
    <div className={cn("space-y-3", className)}>
      {showHeader ? (
        <div>
          <Label className="text-sm font-semibold">Compare with real rovers</Label>
          <p className="text-[0.65rem] leading-snug text-[var(--color-muted-foreground)]">
            Select registry rovers to overlay their evaluator output on the chart
            under the current scenario and mission inputs.
          </p>
        </div>
      ) : null}

      {isPending ? (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Loading rover catalogue…
        </p>
      ) : isError || !data ? (
        <p className="text-xs text-[var(--color-destructive)]">
          Could not load the rover catalogue.
        </p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {data.rovers.map((r) => {
            const selected = overlayRovers.includes(r.rover_name);
            return (
              <button
                key={r.rover_name}
                type="button"
                onClick={() => toggleOverlayRover(r.rover_name)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                  selected
                    ? "border-transparent bg-[rgba(40,75,180,1)] text-white hover:opacity-90"
                    : "border-[var(--color-border)] bg-[var(--color-background)] text-[var(--color-foreground)] hover:bg-[var(--color-accent)]",
                )}
                aria-pressed={selected}
                title={
                  r.is_flown
                    ? `${r.rover_name} (flown mission)`
                    : `${r.rover_name} (design target)`
                }
              >
                {selected ? (
                  <Check className="h-3 w-3" aria-hidden />
                ) : (
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{
                      backgroundColor: roverColor(r.rover_name),
                    }}
                    aria-hidden
                  />
                )}
                {r.rover_name}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );

  if (!framed) return body;

  return (
    <div className="rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/25 px-4 py-3">
      {body}
    </div>
  );
}
