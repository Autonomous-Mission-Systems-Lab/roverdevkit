import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useScenarios } from "@/hooks/use-scenarios";
import { useDesignStore } from "@/store/design-store";
import type { ScenarioName, ScenarioWithSoil } from "@/types/api";

/** Drop-down that picks one of the four canonical scenarios. */
export function ScenarioPicker() {
  const { data, isLoading, isError, error } = useScenarios();
  const scenarioName = useDesignStore((s) => s.scenarioName);
  const setScenario = useDesignStore((s) => s.setScenario);

  const scenarios = data?.scenarios ?? [];
  const selected = scenarios.find(
    (s: ScenarioWithSoil) => s.scenario.name === scenarioName,
  );

  return (
    <div className="space-y-2">
      <Label htmlFor="scenario">Mission scenario</Label>
      <Select
        value={scenarioName}
        onValueChange={(v) => setScenario(v as ScenarioName)}
        disabled={isLoading || isError}
      >
        <SelectTrigger id="scenario" className="w-full">
          <SelectValue
            placeholder={isLoading ? "Loading scenarios..." : "Pick a scenario"}
          />
        </SelectTrigger>
        <SelectContent>
          {scenarios.map((s: ScenarioWithSoil) => (
            <SelectItem key={s.scenario.name} value={s.scenario.name}>
              {humanScenario(s.scenario.name)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {selected ? (
        <p className="text-xs text-[var(--color-muted-foreground)]">
          {selected.scenario.latitude_deg.toFixed(1)}° lat ·{" "}
          {humanText(selected.scenario.terrain_class)} · soil{" "}
          {humanText(selected.soil.simulant)}
        </p>
      ) : null}
      {isError ? (
        <p className="text-xs text-[var(--color-destructive)]">
          {error instanceof Error ? error.message : "Failed to load scenarios."}
        </p>
      ) : null}
    </div>
  );
}

function humanScenario(name: string): string {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/** Lowercase, underscore-free rendering for free-form labels. */
function humanText(value: string): string {
  return value.replace(/_/g, " ");
}
