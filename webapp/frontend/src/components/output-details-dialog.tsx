import { Info } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { PrimaryTarget } from "@/types/api";
import { TARGET_META } from "@/types/api";

/** Held-out test metrics from reports/surrogate_v9 training run (τ=0.5 head, sorted PI repair). */
const SURROGATE_METRICS: Record<
  PrimaryTarget,
  { medianR2: string; coverage: string }
> = {
  range_km: { medianR2: "0.922", coverage: "96.3%" },
  energy_margin_raw_pct: { medianR2: "0.968", coverage: "88.7%" },
  slope_capability_deg: { medianR2: "0.979", coverage: "95.0%" },
  total_mass_kg: { medianR2: "0.999", coverage: "93.8%" },
};

export function OutputDetailsButton({ target }: { target: PrimaryTarget }) {
  const meta = TARGET_META[target];
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          aria-label={`${meta.label} calculation details`}
          className="inline-flex h-5 w-5 items-center justify-center rounded-full text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]"
        >
          <Info className="h-3.5 w-3.5" aria-hidden />
        </button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{meta.label}</DialogTitle>
          <DialogDescription>{meta.description}</DialogDescription>
        </DialogHeader>

        <div className="space-y-5 text-sm leading-6">
          <TargetCalculation target={target} />
          <SurrogatePerformance target={target} />
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TargetCalculation({ target }: { target: PrimaryTarget }) {
  if (target === "range_km") {
    return (
      <section className="space-y-2">
        <h3 className="font-medium">How range is calculated</h3>
        <p className="text-[var(--color-muted-foreground)]">
          The evaluator time-steps the rover over the selected mission scenario.
          At each step, cruise speed is derived from drivetrain torque capacity,
          the slip-balance solution, and available power. Forward progress is
          integrated until the scenario distance cap or mission time is reached.
        </p>
        <Equation>range_km = max(position_m(t_end)) / 1000</Equation>
        <Equation>dx = v_cruise · δ_eff · dt</Equation>
        <Equation>
          if SOC = SOC_min and P_solar &lt; P_avionics + P_mobility, then δ_eff is
          throttled to max(0, (P_solar - P_avionics) / P_mobility)
        </Equation>
        <p className="text-[var(--color-muted-foreground)]">
          Mobility power depends on wheel torque and sinkage, so range
          inherits wheel geometry, soil, mass, slope, solar-area, battery, and
          avionics effects.
        </p>
      </section>
    );
  }
  if (target === "energy_margin_raw_pct") {
    return (
      <section className="space-y-2">
        <h3 className="font-medium">How energy margin is calculated</h3>
        <p className="text-[var(--color-muted-foreground)]">
          Solar generation and electrical loads are integrated over the traverse.
          This raw margin is intentionally unclipped, so negative values mean the
          rover consumed more energy than it generated, while large positive
          values indicate surplus generation.
        </p>
        <Equation>E_gen = ∫ P_solar(t) dt</Equation>
        <Equation>
          E_used = ∫ (P_avionics + P_payload + P_mobility / η_motor) dt
        </Equation>
        <Equation>energy_margin_raw_pct = 100 · (E_gen - E_used) / E_used</Equation>
        <p className="text-[var(--color-muted-foreground)]">
          Solar power depends on scenario latitude, mission duration, panel area,
          panel efficiency, and dust factor. Mobility load comes from the same
          Bekker-Wong wheel-force path used by range; payload power adds to the
          continuous base load. Runs assume a fixed mission start at local
          sunrise (zero solar declination); mission start phase and season are
          held constant rather than swept.
        </p>
      </section>
    );
  }
  if (target === "slope_capability_deg") {
    return (
      <section className="space-y-2">
        <h3 className="font-medium">How slope capability is calculated</h3>
        <p className="text-[var(--color-muted-foreground)]">
          The evaluator searches for the steepest slope where the Bekker-Wong
          wheel-soil model can still generate enough drawbar pull to balance the
          downslope component of rover weight without exceeding available wheel
          torque.
        </p>
        <Equation>
          DP* = m_total · g · sin(θ) / N_w (per wheel)
        </Equation>
        <Equation>
          feasible if F_drawbar(slip, wheel, soil, load) ≥ DP*
        </Equation>
        <Equation>slope_capability_deg = max feasible θ (capped at 35°)</Equation>
        <p className="text-[var(--color-muted-foreground)]">
          Grousers increase shear thrust through an engaged-grouser prefactor
          γ_g = 1 + min(N_g h_g / (2πr), 0.6) applied to the contact shear
          stress.
        </p>
      </section>
    );
  }
  return (
    <section className="space-y-2">
      <h3 className="font-medium">How total mass is calculated</h3>
      <p className="text-[var(--color-muted-foreground)]">
        Total mass is a bottom-up subsystem buildup. The user-provided chassis
        mass anchors the rover bus; wheels, drivetrain, solar, battery, and
        avionics are sized from design inputs, then harness, thermal-control,
        and dry-mass growth fractions are applied. Scientific payload mass from
        the mission scenario is added afterward.
      </p>
      <Equation>
        m_sub = m_chassis + m_wheels + m_motors + m_solar + m_battery + m_avionics
      </Equation>
      <Equation>
        m_dry = m_sub + f_h m_sub + f_t (m_sub + f_h m_sub)
      </Equation>
      <Equation>m_total = m_dry + f_g m_dry + m_payload</Equation>
      <Equation>m_battery = battery_capacity_wh / specific_energy_wh_per_kg</Equation>
      <Equation>
        m_motors = N_w (m_m0 + k_τ · peak_wheel_torque_nm)
      </Equation>
      <p className="text-[var(--color-muted-foreground)]">
        Total mass does not use the wheel-force model directly, but it feeds
        back into mobility because heavier designs increase normal load, sinkage,
        and required drawbar pull.
      </p>
    </section>
  );
}

function SurrogatePerformance({ target }: { target: PrimaryTarget }) {
  const perf = SURROGATE_METRICS[target];
  return (
    <section>
      <h3 className="font-medium">Prediction interval model performance</h3>
      <p className="text-[var(--color-muted-foreground)]">
        The displayed q05–q95 interval comes from quantile gradient-boosted
        heads trained on a 40,000-row stratified Latin-hypercube sample over
        the joint design × scenario space. The median shown in the table is the
        deterministic evaluator output; the surrogate supplies the uncertainty
        envelope around it (and the SHAP attributions on the Explain Design
        tab). These metrics measure emulator fidelity to the analytical
        evaluator, not physical accuracy. The Optimize Design tab&apos;s NSGA-II
        search uses the analytical physics evaluator directly as its fitness
        function.
      </p>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
        <Metric label="Median R²" value={perf.medianR2} />
        <Metric label="90% PI coverage" value={perf.coverage} />
      </div>
    </section>
  );
}

function Equation({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md bg-[var(--color-muted)] px-3 py-2 font-mono text-xs">
      {children}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border px-3 py-2">
      <div className="text-[var(--color-muted-foreground)]">{label}</div>
      <div className="font-semibold tabular-nums">{value}</div>
    </div>
  );
}
