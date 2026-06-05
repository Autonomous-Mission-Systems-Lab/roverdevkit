/**
 * Tiny typed fetch client for the FastAPI backend.
 *
 * Each function maps 1:1 to a backend route in
 * `webapp/backend/routes/`. The base URL is empty by default so the
 * calls are relative — Vite's dev proxy forwards them to
 * http://localhost:8000 (see `vite.config.ts`), and a built bundle
 * served from the same FastAPI server gets them for free. Override
 * via `VITE_API_BASE` for split-host deployments.
 */

import type {
  EvaluateRequest,
  EvaluateResponse,
  HealthResponse,
  OptimizeCancelResponse,
  OptimizeJobResponse,
  OptimizeRequest,
  OptimizeResultResponse,
  PredictRequest,
  PredictResponse,
  RegistryListResponse,
  ScenarioListResponse,
  ShapExplainRequest,
  ShapLocalResponse,
  SweepRequest,
  SweepResponse,
  VersionResponse,
} from "@/types/api";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "") as string;

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = apiUrl(path);
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const response = await fetch(url, { ...init, headers });
  const text = await response.text();
  const body: unknown = text ? safeJson(text) : null;
  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new ApiError(
      response.status,
      detail ?? `HTTP ${response.status} on ${path}`,
      body,
    );
  }
  return body as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export const api = {
  healthz: () => request<HealthResponse>("/healthz"),
  version: () => request<VersionResponse>("/version"),
  listScenarios: () => request<ScenarioListResponse>("/scenarios"),
  listRegistry: () => request<RegistryListResponse>("/registry"),
  predict: (req: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  evaluate: (req: EvaluateRequest) =>
    request<EvaluateResponse>("/evaluate", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  sweep: (req: SweepRequest) =>
    request<SweepResponse>("/sweep", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  optimize: (req: OptimizeRequest) =>
    request<OptimizeJobResponse>("/optimize", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  optimizeResult: (pathOrJobId: string) =>
    request<OptimizeResultResponse>(
      pathOrJobId.startsWith("/") ? pathOrJobId : `/optimize/${pathOrJobId}/result`,
    ),
  cancelOptimize: (pathOrJobId: string) =>
    request<OptimizeCancelResponse>(
      pathOrJobId.startsWith("/") ? pathOrJobId : `/optimize/${pathOrJobId}/cancel`,
      { method: "POST" },
    ),
  shapExplain: (req: ShapExplainRequest) =>
    request<ShapLocalResponse>("/shap/explain", {
      method: "POST",
      body: JSON.stringify(req),
    }),
};

export { ApiError };
