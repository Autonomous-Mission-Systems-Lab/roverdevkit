"""Tests for the soil-simulant catalogue loader."""

from __future__ import annotations

import pytest

from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.soils import (
    SOIL_CSV_PATH,
    get_soil_parameters,
    list_soil_simulants,
    load_soil_catalogue,
)


def test_csv_file_is_present() -> None:
    assert SOIL_CSV_PATH.exists(), f"expected soil CSV at {SOIL_CSV_PATH}"


def test_catalogue_contains_expected_simulants() -> None:
    names = list_soil_simulants()
    # Every scenario YAML references one of these; breaking this test means
    # some scenario cannot be loaded.
    for required in (
        "Apollo_regolith_nominal",
        "Apollo_regolith_loose",
        "Apollo_regolith_dense",
    ):
        assert required in names, f"missing simulant {required!r} in catalogue"


def test_parameters_are_physically_plausible() -> None:
    for name in list_soil_simulants():
        params = get_soil_parameters(name)
        assert 0.5 <= params.n <= 1.5, f"{name}: sinkage exponent out of range"
        # KLS-1 carries a negative k_c from a Wong 1980 least-squares fit;
        # the catalogue notes k_eff = k_c/b + k_phi stays positive at wheel widths.
        if name != "KLS-1":
            assert params.k_c >= 0.0
        assert params.k_phi > 0.0
        assert params.cohesion_kpa >= 0.0
        assert 25.0 <= params.friction_angle_deg <= 55.0
        assert params.shear_modulus_k_m > 0.0


def test_lookup_returns_soil_parameters_type() -> None:
    params = get_soil_parameters("Apollo_regolith_nominal")
    assert isinstance(params, SoilParameters)


def test_unknown_simulant_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError, match="unknown soil simulant"):
        get_soil_parameters("NotARealSimulant")


def test_loader_is_cached_returns_same_dict() -> None:
    a = load_soil_catalogue()
    b = load_soil_catalogue()
    assert a is b
