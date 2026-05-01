"""NSGA-II optimization job routes.

``POST /optimize`` queues a background NSGA-II run, ``/stream`` exposes
per-generation checkpoints as server-sent events, ``/result`` returns
the final Pareto front, and ``/cancel`` requests cooperative
termination. Jobs are intentionally process-local: this is a local-first
tool, and a future deployed queue can preserve the HTTP contract.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Literal

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from roverdevkit.tradespace.optimizer import (
    NSGA2Runner,
    OptimizationCheckpoint,
    OptimizationConstraint,
    OptimizationObjective,
    OptimizationResult,
)
from webapp.backend.loaders import (
    get_canonical_scenarios,
    get_correction,
    get_quantile_bundles,
    get_soil_for_simulant,
)
from webapp.backend.schemas import (
    OptimizeCancelResponse,
    OptimizeCheckpointOut,
    OptimizeJobResponse,
    OptimizeParetoPoint,
    OptimizeRequest,
    OptimizeResultResponse,
)

router = APIRouter(tags=["optimize"])

JOB_TTL_SECONDS = 30 * 60
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rdk-optimize")

JobStatus = Literal["queued", "running", "completed", "cancelled", "failed"]


@dataclass
class _OptimizeJob:
    job_id: str
    status: JobStatus = "queued"
    checkpoints: list[OptimizationCheckpoint] = field(default_factory=list)
    result: OptimizationResult | None = None
    error: str | None = None
    cancel_requested: bool = False
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock)


class _JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, _OptimizeJob] = {}
        self._lock = Lock()

    def create(self) -> _OptimizeJob:
        self.prune()
        job = _OptimizeJob(job_id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> _OptimizeJob:
        self.prune()
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def prune(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if now - job.updated_at > JOB_TTL_SECONDS
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)


_STORE = _JobStore()


@router.post("/optimize", response_model=OptimizeJobResponse)
def optimize(req: OptimizeRequest) -> OptimizeJobResponse:
    """Queue an NSGA-II optimization job and return its job URLs."""
    scenarios = get_canonical_scenarios()
    if req.scenario_name not in scenarios:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown scenario {req.scenario_name!r}. "
                f"Pick one of {sorted(scenarios.keys())}."
            ),
        )
    scenario = scenarios[req.scenario_name]
    if req.operational_duty_cycle is not None:
        scenario = scenario.model_copy(
            update={"operational_duty_cycle": req.operational_duty_cycle}
        )

    soil = get_soil_for_simulant(scenario.soil_simulant)
    correction = get_correction()
    bundles = None
    if req.backend == "surrogate":
        try:
            bundles = get_quantile_bundles()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="surrogate artifact not loaded; run scripts/calibrate_intervals.py first.",
            ) from exc

    objectives = tuple(
        OptimizationObjective(item.target, item.direction) for item in req.objectives
    )
    constraints = tuple(
        OptimizationConstraint(item.target, item.sense, item.value)
        for item in req.constraints
    )

    try:
        runner = NSGA2Runner(
            scenario,
            soil,
            bundles=bundles,
            correction=correction,
            backend=req.backend,
            objectives=objectives,
            constraints=constraints,
            population_size=req.population_size,
            n_generations=req.n_generations,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job = _STORE.create()
    _EXECUTOR.submit(_run_job, job, runner)
    return OptimizeJobResponse(
        job_id=job.job_id,
        status=job.status,
        stream_url=f"/optimize/{job.job_id}/stream",
        result_url=f"/optimize/{job.job_id}/result",
        cancel_url=f"/optimize/{job.job_id}/cancel",
    )


@router.get("/optimize/{job_id}/stream")
async def stream(job_id: str) -> EventSourceResponse:
    """Stream checkpoints for an optimization job as SSE events."""
    job = _lookup_job(job_id)

    async def events():
        sent = 0
        while True:
            with job.lock:
                checkpoints = list(job.checkpoints)
                status = job.status
                error = job.error
                job.updated_at = time.monotonic()

            for checkpoint in checkpoints[sent:]:
                sent += 1
                yield {
                    "event": "checkpoint",
                    "data": _checkpoint_out(checkpoint).model_dump_json(),
                }

            if status in {"completed", "cancelled", "failed"}:
                yield {
                    "event": status,
                    "data": json.dumps({"job_id": job.job_id, "status": status, "error": error}),
                }
                break
            await asyncio.sleep(0.2)

    return EventSourceResponse(events())


@router.get("/optimize/{job_id}/result", response_model=OptimizeResultResponse)
def result(job_id: str) -> OptimizeResultResponse:
    """Return job state and the final Pareto front when available."""
    job = _lookup_job(job_id)
    with job.lock:
        return _result_response(job)


@router.post("/optimize/{job_id}/cancel", response_model=OptimizeCancelResponse)
def cancel(job_id: str) -> OptimizeCancelResponse:
    """Request cooperative cancellation of a queued or running job."""
    job = _lookup_job(job_id)
    with job.lock:
        if job.status in {"queued", "running"}:
            job.cancel_requested = True
            job.updated_at = time.monotonic()
        return OptimizeCancelResponse(job_id=job.job_id, status=job.status)


def _run_job(job: _OptimizeJob, runner: NSGA2Runner) -> None:
    def on_checkpoint(checkpoint: OptimizationCheckpoint) -> None:
        with job.lock:
            job.checkpoints.append(checkpoint)
            job.updated_at = time.monotonic()

    def should_cancel() -> bool:
        with job.lock:
            return job.cancel_requested

    with job.lock:
        job.status = "running"
        job.updated_at = time.monotonic()
    try:
        result = runner.run(on_checkpoint=on_checkpoint, should_cancel=should_cancel)
    except Exception as exc:  # pragma: no cover - surfaced via API
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = time.monotonic()
        return
    with job.lock:
        job.result = result
        job.status = "cancelled" if job.cancel_requested else "completed"
        job.updated_at = time.monotonic()


def _lookup_job(job_id: str) -> _OptimizeJob:
    try:
        return _STORE.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown optimization job {job_id!r}") from exc


def _checkpoint_out(checkpoint: OptimizationCheckpoint) -> OptimizeCheckpointOut:
    return OptimizeCheckpointOut(
        gen=checkpoint.gen,
        hypervolume=checkpoint.hypervolume,
        pareto_size=checkpoint.pareto_size,
        best_per_objective=checkpoint.best_per_objective,
    )


def _result_response(job: _OptimizeJob) -> OptimizeResultResponse:
    checkpoints = [_checkpoint_out(checkpoint) for checkpoint in job.checkpoints]
    pareto_front: list[OptimizeParetoPoint] = []
    backend_used: Literal["surrogate", "evaluator"] | None = None
    if job.result is not None:
        backend_used = job.result.backend_used
        pareto_front = [
            OptimizeParetoPoint(design=design, metrics=metrics)
            for design, metrics in zip(
                job.result.design_vectors,
                job.result.metrics,
                strict=True,
            )
        ]
    return OptimizeResultResponse(
        job_id=job.job_id,
        status=job.status,
        backend_used=backend_used,
        checkpoints=checkpoints,
        pareto_front=pareto_front,
        error=job.error,
    )
