import { useMutation, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ShapExplainRequest, ShapLocalResponse } from "@/types/api";

/** Global median-head feature importance for all primary targets. */
export function useShapGlobal() {
  return useQuery({
    queryKey: ["shap", "global"],
    queryFn: () => api.shapGlobal(),
    staleTime: 5 * 60 * 1000,
  });
}

/** Per-design TreeSHAP-style contributions for a selected target. */
export function useShapExplain() {
  return useMutation<ShapLocalResponse, Error, ShapExplainRequest>({
    mutationKey: ["shap", "explain"],
    mutationFn: (req) => api.shapExplain(req),
  });
}
