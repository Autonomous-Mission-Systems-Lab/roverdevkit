import { create } from "zustand";

import {
  DESIGN_BOUNDS,
  SWEEPABLE_VARIABLES,
  type PrimaryTarget,
  type SweepBackend,
  type SweepableVariable,
} from "@/types/api";

/**
 * Local UI state for the sweep page.
 *
 * Kept separate from the single-design store because the two pages
 * have different "do you want this to persist when I tab away" intent:
 * the single-design panel preserves the candidate across tab switches
 * (you're iterating on it), while the sweep page should snap back to
 * sane defaults when the user revisits it.
 */

interface AxisDraft {
  variable: SweepableVariable;
  lo: number;
  hi: number;
  n_points: number;
}

interface SweepState {
  target: PrimaryTarget;
  xAxis: AxisDraft;
  yAxis: AxisDraft | null;
  backend: SweepBackend;

  setTarget: (target: PrimaryTarget) => void;
  setXAxis: (axis: Partial<AxisDraft>) => void;
  setXVariable: (variable: SweepableVariable) => void;
  setYEnabled: (enabled: boolean) => void;
  setYAxis: (axis: Partial<AxisDraft>) => void;
  setYVariable: (variable: SweepableVariable) => void;
  setBackend: (backend: SweepBackend) => void;
  reset: () => void;
}

/** Pull the schema bounds for `variable` and a sensible n-points default. */
export function defaultAxisFor(variable: SweepableVariable): AxisDraft {
  const bounds = DESIGN_BOUNDS[variable];
  return {
    variable,
    lo: bounds.min,
    hi: bounds.max,
    n_points: variable === "grouser_count" ? 7 : 11,
  };
}

const DEFAULT_X: SweepableVariable = "wheel_radius_m";
const DEFAULT_TARGET: PrimaryTarget = "range_km";

const DEFAULTS = {
  target: DEFAULT_TARGET,
  xAxis: defaultAxisFor(DEFAULT_X),
  yAxis: null as AxisDraft | null,
  backend: "auto" as SweepBackend,
};

export const useSweepStore = create<SweepState>()((set, get) => ({
  ...DEFAULTS,
  setTarget: (target) => set({ target }),
  setXAxis: (axis) => set((state) => ({ xAxis: { ...state.xAxis, ...axis } })),
  setXVariable: (variable) => {
    // Picking a new x variable resets bounds to that field's
    // schema range so the lo/hi defaults always sit inside it.
    const next = defaultAxisFor(variable);
    // If the y axis already uses this variable, swap it to a
    // different sweepable so the spec stays valid.
    const y = get().yAxis;
    if (y && y.variable === variable) {
      const fallback = SWEEPABLE_VARIABLES.find((v) => v !== variable);
      set({
        xAxis: next,
        yAxis: fallback ? defaultAxisFor(fallback) : null,
      });
    } else {
      set({ xAxis: next });
    }
  },
  setYEnabled: (enabled) =>
    set((state) => {
      if (!enabled) return { yAxis: null };
      if (state.yAxis) return state;
      const fallback =
        SWEEPABLE_VARIABLES.find((v) => v !== state.xAxis.variable) ??
        SWEEPABLE_VARIABLES[0];
      return { yAxis: defaultAxisFor(fallback) };
    }),
  setYAxis: (axis) =>
    set((state) =>
      state.yAxis
        ? { yAxis: { ...state.yAxis, ...axis } }
        : { yAxis: state.yAxis },
    ),
  setYVariable: (variable) => {
    const next = defaultAxisFor(variable);
    set({ yAxis: next });
  },
  setBackend: (backend) => set({ backend }),
  reset: () => set({ ...DEFAULTS }),
}));
