from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _payload(*, population_size: int = 4, n_generations: int = 1) -> dict[str, object]:
    return {
        "scenario_name": "equatorial_mare_traverse",
        "backend": "evaluator",
        "population_size": population_size,
        "n_generations": n_generations,
        "seed": 3,
        "objectives": [
            {"target": "range_km", "direction": "max"},
            {"target": "total_mass_kg", "direction": "min"},
        ],
        "constraints": [
            {"target": "slope_capability_deg", "sense": "min", "value": 5.0},
        ],
    }


def test_optimize_evaluator_job_completes_and_returns_front(client: TestClient) -> None:
    response = client.post("/optimize", json=_payload())
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] in {"queued", "running"}

    result = None
    for _ in range(30):
        result_response = client.get(job["result_url"])
        assert result_response.status_code == 200, result_response.text
        result = result_response.json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.25)

    assert result is not None
    assert result["status"] == "completed", result
    assert result["backend_used"] == "evaluator"
    assert result["checkpoints"]
    assert result["pareto_front"]
    first = result["pareto_front"][0]
    assert "design" in first
    assert "range_km" in first["metrics"]


def test_optimize_accepts_required_obstacle_height_override(client: TestClient) -> None:
    payload = _payload(population_size=4, n_generations=1)
    payload["required_obstacle_height_m"] = 0.10
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200, response.text


def test_optimize_evaluator_budget_cap_returns_422(client: TestClient) -> None:
    # The webapp optimize route sets evaluator_eval_cap=5000 on the
    # NSGA2Runner so a worst-case live job finishes inside ~2 min wall
    # clock at the corrected evaluator's ~22 ms/call. Anything beyond
    # that returns 422 with a message referencing the cap.
    response = client.post(
        "/optimize",
        json=_payload(population_size=200, n_generations=50),
    )
    assert response.status_code == 422, response.text
    assert "capped at 5000 evaluations" in response.json()["detail"]


def test_optimize_unknown_scenario_returns_404(client: TestClient) -> None:
    payload = _payload()
    payload["scenario_name"] = "not_a_real_scenario"
    response = client.post("/optimize", json=payload)
    assert response.status_code == 404, response.text
