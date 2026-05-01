import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

import { AppShell } from "@/components/app-shell";
import { DesignExplorer } from "@/pages/design-explorer";
import { ParetoCompute } from "@/pages/pareto-compute";
import { ParametricSweep } from "@/pages/parametric-sweep";
import { ShapRules } from "@/pages/shap-rules";
import { useViewStore } from "@/store/view-store";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>
        <CurrentView />
      </AppShell>
      {import.meta.env.DEV ? <ReactQueryDevtools position="bottom" /> : null}
    </QueryClientProvider>
  );
}

function CurrentView() {
  const view = useViewStore((s) => s.view);
  if (view === "shap") return <ShapRules />;
  if (view === "pareto") return <ParetoCompute />;
  if (view === "sweep") return <ParametricSweep />;
  return <DesignExplorer />;
}
