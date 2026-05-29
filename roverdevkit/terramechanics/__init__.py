"""Terramechanics sub-models.

- :mod:`.bekker_wong` — Bekker-Wong pressure-sinkage with Janosi-Hanamoto
  shear. Fast analytical path (Path 1). Re-exported here for convenience.
- :mod:`.pychrono_scm` — single-wheel PyChrono Soil Contact Model driver,
  drop-in signature analog of :func:`.bekker_wong.single_wheel_forces`
  with the entry point :func:`.pychrono_scm.single_wheel_forces_scm`.
  **Not** re-exported at package level: importing the module triggers
  the OpenMP preload shim and ``import pychrono``, which adds ~350 ms
  to package init that the analytical path does not need. Callers
  use the explicit import path.
- :mod:`.soils` — name -> :class:`SoilParameters` lookup backed by
  :file:`data/soil_simulants.csv`.
- :mod:`.correction_model` — SCM-correction gate ML correction layer that predicts
  the per-wheel delta between SCM and Bekker-Wong over the wheel-level
  feature space used by the correction-model pipeline.

The SCM-correction batch orchestration (parallel SCM sweep, resumable queue,
parquet I/O) lives in ``scripts/`` rather than the importable package
to keep the package light for analytical-only consumers.
"""

from roverdevkit.terramechanics.bekker_wong import (
    SoilParameters,
    WheelForces,
    WheelGeometry,
    single_wheel_forces,
)
from roverdevkit.terramechanics.soils import (
    SoilSimulantRecord,
    get_soil_parameters,
    list_soil_simulants,
    load_soil_catalogue,
)

__all__ = [
    "SoilParameters",
    "SoilSimulantRecord",
    "WheelForces",
    "WheelGeometry",
    "get_soil_parameters",
    "list_soil_simulants",
    "load_soil_catalogue",
    "single_wheel_forces",
]
