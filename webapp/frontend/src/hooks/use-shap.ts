import { useMutation } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { ShapExplainRequest, ShapLocalResponse } from "@/types/api";

/** Per-design TreeSHAP-style contributions for a selected target. */
export function useShapExplain() {
  return useMutation<ShapLocalResponse, Error, ShapExplainRequest>({
    mutationKey: ["shap", "explain"],
    mutationFn: (req) => api.shapExplain(req),
  });
}
