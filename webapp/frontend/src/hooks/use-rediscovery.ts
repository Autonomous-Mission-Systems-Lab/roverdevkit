import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { RediscoveryBackend } from "@/types/api";

/**
 * Lazily fetched rediscovery LOO summary for one backend.
 *
 * The artifact is a committed JSON tree under
 * `reports/rediscovery_loo_*` and is immutable for the lifetime of
 * the backend process; cache forever and never refetch on focus.
 */
export function useRediscoveryList(backend: RediscoveryBackend) {
  return useQuery({
    queryKey: ["rediscovery", "list", backend],
    queryFn: () => api.listRediscovery(backend),
    staleTime: Infinity,
    gcTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

/**
 * Per-rover rediscovery detail. The Pareto front payload can be
 * large (~500 points × 11 design fields), so we cache aggressively
 * and gate the query on a non-null slug so the page can render the
 * "no rover selected" state without a wasted round-trip.
 */
export function useRediscoveryDetail(
  slug: string | null,
  backend: RediscoveryBackend,
) {
  return useQuery({
    queryKey: ["rediscovery", "detail", backend, slug],
    queryFn: () => {
      if (!slug) throw new Error("rediscovery slug is required");
      return api.getRediscoveryDetail(slug, backend);
    },
    enabled: slug !== null,
    staleTime: Infinity,
    gcTime: 60 * 60 * 1000,
    refetchOnWindowFocus: false,
  });
}
