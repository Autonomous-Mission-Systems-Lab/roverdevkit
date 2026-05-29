import { create } from "zustand";

import type {
  OptimizeObjectiveIn,
  OptimizeParetoPoint,
  OptimizeResultResponse,
} from "@/types/api";

export interface ActiveParetoFront {
  label: string;
  scenarioName: string;
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
  clearFront: () => void;
}

export const useParetoStore = create<ParetoState>()((set) => ({
  activeFront: null,
  setFromOptimizeResult: (result, scenarioName, objectives) =>
    set({
      activeFront: {
        label: `Custom run · ${scenarioName}`,
        scenarioName,
        points: result.pareto_front,
        metadata: {
          job_id: result.job_id,
          backend_used: result.backend_used,
          status: result.status,
          objectives,
        },
      },
    }),
  clearFront: () => set({ activeFront: null }),
}));
