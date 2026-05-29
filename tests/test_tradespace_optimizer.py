from __future__ import annotations

import numpy as np
import pandas as pd

from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.tradespace.optimizer import NSGA2Runner


class _FakeHead:
    def __init__(self, target: str) -> None:
        self.target = target

    def predict(self, X: pd.DataFrame, *, repair_crossings: bool = True) -> dict[str, np.ndarray]:
        if self.target == "range_km":
            q50 = 10.0 * X["design_solar_area_m2"] + X["design_wheel_radius_m"]
        elif self.target == "energy_margin_raw_pct":
            q50 = 100.0 * X["design_solar_area_m2"] - X["design_avionics_power_w"]
        elif self.target == "slope_capability_deg":
            q50 = 2.0 * X["design_peak_wheel_torque_nm"]
        elif self.target == "total_mass_kg":
            q50 = X["design_chassis_mass_kg"] + 0.01 * X["design_battery_capacity_wh"]
        else:  # pragma: no cover - test fixture target list is fixed
            raise KeyError(self.target)
        values = np.asarray(q50, dtype=float)
        return {"q05": values - 1.0, "q50": values, "q95": values + 1.0}


def test_nsga2_runner_surrogate_emits_checkpoints_and_front() -> None:
    scenario = load_scenario("equatorial_mare_traverse")
    soil = get_soil_parameters(scenario.soil_simulant)
    bundles = {
        target: _FakeHead(target)
        for target in (
            "range_km",
            "energy_margin_raw_pct",
            "slope_capability_deg",
            "total_mass_kg",
        )
    }

    result = NSGA2Runner(
        scenario,
        soil,
        bundles=bundles,  # type: ignore[arg-type]
        backend="surrogate",
        population_size=8,
        n_generations=2,
        seed=1,
    ).run()

    assert result.backend_used == "surrogate"
    assert result.checkpoints
    assert result.design_vectors
    assert result.metrics
    assert all(0.05 <= d.wheel_radius_m <= 0.20 for d in result.design_vectors)
    assert all(d.n_wheels in {4, 6} for d in result.design_vectors)
    assert all("range_km" in row for row in result.metrics)
