import { useMutation } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  OptimizeCancelResponse,
  OptimizeJobResponse,
  OptimizeRequest,
  OptimizeResultResponse,
} from "@/types/api";

/** `POST /optimize` mutation that queues an NSGA-II job. */
export function useStartOptimize() {
  return useMutation<OptimizeJobResponse, Error, OptimizeRequest>({
    mutationKey: ["optimize", "start"],
    mutationFn: (req) => api.optimize(req),
  });
}

/** Fetch the final/current state for an optimization job. */
export function useOptimizeResult() {
  return useMutation<OptimizeResultResponse, Error, string>({
    mutationKey: ["optimize", "result"],
    mutationFn: (pathOrJobId) => api.optimizeResult(pathOrJobId),
  });
}

/** Request cooperative cancellation for an optimization job. */
export function useCancelOptimize() {
  return useMutation<OptimizeCancelResponse, Error, string>({
    mutationKey: ["optimize", "cancel"],
    mutationFn: (pathOrJobId) => api.cancelOptimize(pathOrJobId),
  });
}
