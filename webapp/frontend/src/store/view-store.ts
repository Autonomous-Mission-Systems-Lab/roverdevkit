import { create } from "zustand";

export type AppView = "design" | "sweep" | "pareto" | "shap" | "validate";

interface ViewState {
  view: AppView;
  setView: (view: AppView) => void;
}

/**
 * Top-level navigation state.
 *
 * Trivial Zustand store rather than a router because the whole app
 * is currently a 2-tab interface and pulling in `react-router-dom`
 * for that would be over-kill. If we add per-route URLs (sharable
 * deep links into a sweep config) we'll switch to a real router.
 */
export const useViewStore = create<ViewState>()((set) => ({
  view: "design",
  setView: (view) => set({ view }),
}));
