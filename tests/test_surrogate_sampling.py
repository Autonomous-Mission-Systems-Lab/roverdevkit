"""Unit tests for the stratified LHS sampler (initial baseline-surrogate schema)."""

from __future__ import annotations

from collections import Counter

import pytest

from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.surrogate.sampling import (
    _CONTINUOUS_DESIGN_BOUNDS,
    _GROUSER_COUNT_BOUNDS,
    _SOIL_BOUNDS,
    FAMILIES,
    LHSSample,
    generate_samples,
)

# ---------------------------------------------------------------------------
# Basic shape / typing
# ---------------------------------------------------------------------------


def test_generate_samples_total_count_matches_contract() -> None:
    samples = generate_samples(n_per_scenario=8, seed=0)
    assert len(samples) == 8 * len(FAMILIES)


def test_generate_samples_subset_of_families() -> None:
    samples = generate_samples(
        n_per_scenario=4,
        seed=0,
        scenario_names=["equatorial_mare_traverse", "polar_prospecting"],
    )
    assert len(samples) == 8
    families = {s.scenario_family for s in samples}
    assert families == {"equatorial_mare_traverse", "polar_prospecting"}


def test_generate_samples_rejects_unknown_family() -> None:
    with pytest.raises(KeyError, match="unknown scenario family"):
        generate_samples(n_per_scenario=2, scenario_names=["no_such_scenario"])


def test_generate_samples_rejects_odd_n() -> None:
    with pytest.raises(ValueError, match="even"):
        generate_samples(n_per_scenario=7)


def test_generate_samples_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_samples(n_per_scenario=0)


def test_sample_objects_are_typed() -> None:
    samples = generate_samples(n_per_scenario=2, seed=0)
    assert all(isinstance(s, LHSSample) for s in samples)
    assert all(isinstance(s.design, DesignVector) for s in samples)
    assert all(isinstance(s.scenario, MissionScenario) for s in samples)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generate_samples_is_deterministic_for_same_seed() -> None:
    s1 = generate_samples(n_per_scenario=8, seed=123)
    s2 = generate_samples(n_per_scenario=8, seed=123)
    for a, b in zip(s1, s2, strict=True):
        assert a.design == b.design
        assert a.scenario == b.scenario
        assert a.soil == b.soil
        assert a.split == b.split
        assert a.sample_index == b.sample_index


def test_different_seeds_produce_different_draws() -> None:
    s1 = generate_samples(n_per_scenario=8, seed=123)
    s2 = generate_samples(n_per_scenario=8, seed=456)
    diffs = sum(1 for a, b in zip(s1, s2, strict=True) if a.design != b.design)
    assert diffs > 0


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def test_wheel_strata_are_exact_50_50() -> None:
    samples = generate_samples(n_per_scenario=20, seed=7)
    for family_name in FAMILIES:
        fam = [s for s in samples if s.scenario_family == family_name]
        counts = Counter(s.design.n_wheels for s in fam)
        assert counts[4] == 10, f"{family_name}: 4-wheel count {counts[4]} != 10"
        assert counts[6] == 10, f"{family_name}: 6-wheel count {counts[6]} != 10"


def test_stratum_id_matches_n_wheels() -> None:
    samples = generate_samples(n_per_scenario=8, seed=0)
    for s in samples:
        expected = 0 if s.design.n_wheels == 4 else 1
        assert s.stratum_id == expected


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_split_labels_are_valid() -> None:
    samples = generate_samples(n_per_scenario=10, seed=0)
    labels = {s.split for s in samples}
    assert labels <= {"train", "val", "test"}


def test_split_fractions_roughly_match_request() -> None:
    samples = generate_samples(n_per_scenario=200, seed=0, val_frac=0.2, test_frac=0.1)
    counts = Counter(s.split for s in samples)
    total = len(samples)
    train_frac = counts["train"] / total
    val_frac = counts["val"] / total
    test_frac = counts["test"] / total
    assert abs(train_frac - 0.7) < 0.05
    assert abs(val_frac - 0.2) < 0.05
    assert abs(test_frac - 0.1) < 0.05


def test_invalid_split_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        generate_samples(n_per_scenario=4, val_frac=-0.1)
    with pytest.raises(ValueError):
        generate_samples(n_per_scenario=4, val_frac=0.6, test_frac=0.5)


# ---------------------------------------------------------------------------
# Bounds / coverage
# ---------------------------------------------------------------------------


def test_continuous_design_vars_are_within_bounds() -> None:
    samples = generate_samples(n_per_scenario=100, seed=11)
    for name, lo, hi in _CONTINUOUS_DESIGN_BOUNDS:
        values = [getattr(s.design, name) for s in samples]
        assert min(values) >= lo - 1e-9, name
        assert max(values) <= hi + 1e-9, name


def test_grouser_count_is_integer_in_bounds() -> None:
    samples = generate_samples(n_per_scenario=100, seed=11)
    lo, hi = _GROUSER_COUNT_BOUNDS
    for s in samples:
        assert isinstance(s.design.grouser_count, int)
        assert lo <= s.design.grouser_count <= hi


def test_soil_parameters_within_bounds() -> None:
    samples = generate_samples(n_per_scenario=100, seed=11)
    for col, (lo, hi) in _SOIL_BOUNDS.items():
        attr = col[len("soil_") :]
        values = [getattr(s.soil, attr) for s in samples]
        assert min(values) >= lo - 1e-9, col
        assert max(values) <= hi + 1e-9, col


def test_scenario_perturbation_stays_within_family_ranges() -> None:
    samples = generate_samples(n_per_scenario=50, seed=11)
    for s in samples:
        fam = FAMILIES[s.scenario_family]
        assert fam.latitude_range_deg[0] - 1e-9 <= s.scenario.latitude_deg
        assert s.scenario.latitude_deg <= fam.latitude_range_deg[1] + 1e-9
        assert fam.mission_duration_range_days[0] - 1e-9 <= s.scenario.mission_duration_earth_days
        assert s.scenario.mission_duration_earth_days <= fam.mission_duration_range_days[1] + 1e-9
        assert fam.max_slope_range_deg[0] - 1e-9 <= s.scenario.max_slope_deg
        assert s.scenario.max_slope_deg <= fam.max_slope_range_deg[1] + 1e-9
        assert s.scenario.terrain_class == fam.terrain_class
        assert s.scenario.soil_simulant == fam.soil_simulant
        assert s.scenario.sun_geometry == fam.sun_geometry


def test_lhs_covers_design_space_broadly() -> None:
    """A crude coverage sanity check: with 400 samples, the min/max of
    each continuous column should cover at least 80% of the bound range."""
    samples = generate_samples(n_per_scenario=100, seed=3)
    for name, lo, hi in _CONTINUOUS_DESIGN_BOUNDS:
        values = [getattr(s.design, name) for s in samples]
        span = hi - lo
        realised = max(values) - min(values)
        assert realised / span > 0.8, f"{name}: coverage {realised / span:.2%}"


# ---------------------------------------------------------------------------
# Sample indexing
# ---------------------------------------------------------------------------


def test_sample_indices_are_dense_and_ordered() -> None:
    samples = generate_samples(n_per_scenario=6, seed=0)
    assert [s.sample_index for s in samples] == list(range(len(samples)))
