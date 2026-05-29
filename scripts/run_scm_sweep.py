"""SCM-correction SCM single-wheel batch sweep with resumable parquet output.

Generates the (Bekker-Wong, PyChrono SCM) paired single-wheel dataset that
feeds the SCM-correction gate gate (correction-magnitude decision) and, if the gate
fires, the wheel-level correction model.
for the design and budget.

Examples
--------
Smoke run (12 points, 1 worker, fast config — confirms wiring before a
big run)::

    python scripts/run_scm_sweep.py \\
        --n-runs 12 --workers 1 --config fast \\
        --out data/scm/runs_smoke.parquet

Production gate sweep (500 points, 5 workers, default fidelity)::

    python scripts/run_scm_sweep.py \\
        --n-runs 500 --workers 5 \\
        --out data/scm/runs_v1.parquet

Resume an interrupted sweep (skips already-completed ``row_id`` values)::

    python scripts/run_scm_sweep.py \\
        --n-runs 500 --workers 5 \\
        --out data/scm/runs_v1.parquet --resume

Design notes
------------
- Rows are produced by :func:`roverdevkit.terramechanics.scm_sweep.build_design`
  with a stratified-categorical 6-d LHS over the continuous wheel/operating
  parameters and balanced (soil × grouser) buckets. Same ``(n_runs, seed)``
  → same design.
- The pool uses ``multiprocessing`` ``spawn`` context so worker processes
  do a clean PyChrono import (no fork-after-init pitfalls).
- Each completed row is flushed to the parquet store every
  ``--checkpoint-every`` rows so an OS interrupt loses at most that many
  runs of work.
- Failures (BW or SCM) are recorded in-row with status flags; the script
  does not raise and prints a per-status summary at the end.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from roverdevkit.terramechanics.scm_sweep import build_design, run_one

# Pre-canned config presets keyed by the --config flag. Imported lazily
# inside _resolve_scm_config so the CLI parser stays cheap and the module
# is importable without PyChrono present.


def _resolve_scm_config(name: str):
    """Map a config preset name to a :class:`ScmConfig` instance."""
    from roverdevkit.terramechanics.pychrono_scm import ScmConfig

    if name == "default":
        return ScmConfig()
    if name == "fast":
        return ScmConfig(
            time_step_s=1e-3,
            settle_time_s=0.2,
            drive_time_s=0.8,
            terrain_mesh_res_m=0.02,
            average_window_skip=0.5,
        )
    if name == "fine":
        return ScmConfig(
            time_step_s=1e-3,
            settle_time_s=0.3,
            drive_time_s=1.5,
            terrain_mesh_res_m=0.010,
            average_window_skip=0.5,
        )
    raise ValueError(f"unknown SCM config preset {name!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-runs", type=int, required=True, help="number of design points")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output parquet path (parent directories created if needed)",
    )
    p.add_argument("--seed", type=int, default=42, help="design RNG seed")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel workers (use 4-5 on M-series, fewer on CI)",
    )
    p.add_argument(
        "--config",
        choices=("fast", "default", "fine"),
        default="default",
        help="SCM fidelity preset; 'default' is production",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="if --out exists, skip rows whose row_id is already present",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="flush partial results to parquet every N completed rows",
    )
    return p.parse_args(argv)


def _maybe_load_existing(out_path: Path, resume: bool) -> tuple[pd.DataFrame | None, set[int]]:
    """Return previously-completed rows + their row_id set, if resuming."""
    if not resume or not out_path.exists():
        return None, set()
    prev = pd.read_parquet(out_path)
    return prev, set(prev["row_id"].astype(int).tolist())


def _flush(path: Path, frames: list[pd.DataFrame]) -> None:
    """Concatenate frames and write to parquet (atomic via tmp + rename)."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    design = build_design(args.n_runs, seed=args.seed)
    prev_df, completed_ids = _maybe_load_existing(args.out, args.resume)
    todo = design[~design["row_id"].isin(completed_ids)].to_dict(orient="records")

    if completed_ids:
        print(
            f"Resuming: {len(completed_ids)}/{len(design)} rows already complete; "
            f"running {len(todo)} new rows."
        )
    else:
        print(f"Running {len(todo)} SCM points with {args.workers} worker(s).")

    if not todo:
        print("Nothing to do; design is fully covered by the existing parquet.")
        return 0

    scm_config = _resolve_scm_config(args.config)
    print(
        f"SCM config: {args.config!r} "
        f"(settle={scm_config.settle_time_s}s, drive={scm_config.drive_time_s}s, "
        f"mesh={scm_config.terrain_mesh_res_m}m)"
    )

    accumulated = [prev_df] if prev_df is not None else []
    pending: list[dict] = []
    statuses: Counter[str] = Counter()

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()

    if args.workers <= 1:
        # Serial path — useful for CI smoke runs and for keeping logs
        # linear when debugging a worker-side regression.
        for i, row in enumerate(todo):
            res = run_one(row, scm_config=scm_config)
            pending.append(res)
            statuses[res.get("scm_status", "unknown")] += 1
            if (i + 1) % args.checkpoint_every == 0 or i == len(todo) - 1:
                accumulated.append(pd.DataFrame(pending))
                _flush(args.out, accumulated)
                pending = []
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (len(todo) - i - 1)
                print(
                    f"  [{i + 1}/{len(todo)}] checkpointed; elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as exe:
            futures = {
                exe.submit(run_one, row, scm_config=scm_config): row["row_id"] for row in todo
            }
            for i, fut in enumerate(as_completed(futures)):
                res = fut.result()
                pending.append(res)
                statuses[res.get("scm_status", "unknown")] += 1
                if (i + 1) % args.checkpoint_every == 0 or i == len(todo) - 1:
                    accumulated.append(pd.DataFrame(pending))
                    _flush(args.out, accumulated)
                    pending = []
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / (i + 1) * (len(todo) - i - 1)
                    print(
                        f"  [{i + 1}/{len(todo)}] checkpointed; "
                        f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
                    )

    if pending:
        accumulated.append(pd.DataFrame(pending))
        _flush(args.out, accumulated)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {sum(len(d) for d in accumulated)} total rows to {args.out}.")
    print(f"SCM status counts (this run): {dict(statuses)}")
    print(f"Wall-clock: {elapsed:.1f}s on {args.workers} worker(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
