import { useMutation, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ParetoFrontResponse } from "@/types/api";

/** List canonical fronts generated under `reports/phase3_pareto/`. */
export function useParetoFronts() {
  return useQuery({
    queryKey: ["pareto-fronts"],
    queryFn: () => api.listParetoFronts(),
    staleTime: 60 * 1000,
  });
}

/** Load one canonical front by scenario name or API URL. */
export function useLoadParetoFront() {
  return useMutation<ParetoFrontResponse, Error, string>({
    mutationKey: ["pareto-front"],
    mutationFn: (pathOrScenario) => api.paretoFront(pathOrScenario),
  });
}
