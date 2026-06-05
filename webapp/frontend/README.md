# RoverDevKit Frontend

React + TypeScript + Vite single-page app for RoverDevKit. It talks to the
FastAPI backend (see [`../README.md`](../README.md)) for evaluation,
surrogate prediction, parametric sweeps, NSGA-II optimization, and SHAP-style
explanations.

## Stack

- React 19 + TypeScript, built with Vite
- TanStack Query for server state, Zustand for view state
- Tailwind CSS + Radix UI primitives
- Plotly for charts

The typed fetch client lives in `src/lib/api.ts`; the dev server proxies
backend routes to `http://localhost:8000`.

## Develop

```bash
npm install
npm run dev      # Vite dev server on http://localhost:5173
npm run build    # type-check + production build
npm run lint     # ESLint
npm run format   # Prettier
```

Run the backend and frontend together from the repo root with `make webapp-dev`.
