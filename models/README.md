# Models

Shipped trained surrogate bundles used at runtime by the web app and
validation scripts.

| Path | Purpose |
| --- | --- |
| `surrogate_v9/quantile_bundles.joblib` | v9 quantile-XGB heads (calibrated 90% PIs) for the Current Design and Explain Design tabs |

Training-time metrics (`coverage.csv`, `median_sanity.csv`, etc.) are
written to `reports/surrogate_v9/` when you re-fit. A full calibration
run via `scripts/calibrate_intervals.py` publishes the runtime bundle
here automatically:

```bash
python scripts/calibrate_intervals.py \
  --dataset data/analytical/lhs_v9.parquet \
  --tuned-params reports/tuned_v9/tuned_best_params.json
```

Use `--no-publish-bundle` on smoke runs so partial calibrations do not
overwrite the shipped model. Override the runtime path with
`ROVERDEVKIT_QUANTILE_BUNDLES` if needed.
