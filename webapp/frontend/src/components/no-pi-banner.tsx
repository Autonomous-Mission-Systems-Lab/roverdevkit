import { AlertCircle } from "lucide-react";

/**
 * Inline banner shown on the prediction panel whenever the `/predict`
 * route returns `mode = "evaluator_only"`.
 *
 * SCHEMA_VERSION v7_1 (W12 step B follow-on): `operational_duty_cycle`
 * is now a per-row LHS feature, so the live route always returns
 * `"surrogate"` — this banner is dormant in the v7_1 deployment. It
 * is retained as a safety net for any future evaluator-only fallback
 * paths (e.g. out-of-bounds inputs the surrogate refuses to predict
 * on) so the chart doesn't render an empty PI band silently.
 */
export function NoPiBanner() {
  return (
    <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-amber-900 dark:text-amber-300">
      <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <p>
        <span className="font-medium">No prediction interval.</span> The
        request fell back to the deterministic evaluator path, so no calibrated
        90 % band is available for this query.
      </p>
    </div>
  );
}
