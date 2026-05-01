import { create } from "zustand";

import type {
  OptimizeObjectiveIn,
  OptimizeParetoPoint,
  OptimizeResultResponse,
  ParetoFrontResponse,
} from "@/types/api";

export interface ActiveParetoFront {
  label: string;
  scenarioName: string;
  source: "custom" | "canonical";
  points: OptimizeParetoPoint[];
  metadata?: Record<string, unknown>;
}

interface ParetoState {
  activeFront: ActiveParetoFront | null;
  setFromOptimizeResult: (
    result: OptimizeResultResponse,
    scenarioName: string,
    objectives: OptimizeObjectiveIn[],
  ) => void;
  setFromCanonicalFront: (front: ParetoFrontResponse) => void;
  clearFront: () => void;
}

export const useParetoStore = create<ParetoState>()((set) => ({
  activeFront: null,
  setFromOptimizeResult: (result, scenarioName, objectives) =>
    set({
      activeFront: {
        label: `Custom run · ${scenarioName}`,
        scenarioName,
        source: "custom",
        points: result.pareto_front,
        metadata: {
          job_id: result.job_id,
          backend_used: result.backend_used,
          status: result.status,
          objectives,
        },
      },
    }),
  setFromCanonicalFront: (front) =>
    set({
      activeFront: {
        label: `Canonical · ${front.scenario_name}`,
        scenarioName: front.scenario_name,
        source: "canonical",
        points: front.pareto_front,
        metadata: front.metadata,
      },
    }),
  clearFront: () => set({ activeFront: null }),
}));
