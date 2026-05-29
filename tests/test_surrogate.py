"""Light smoke tests for the surrogate sub-package's column inventories.

Detailed dataset/sampler tests live in test_surrogate_sampling.py and
test_surrogate_dataset.py; this module only checks the cross-cutting
column-list invariants exposed by features.py so a stale rename is
caught at the smallest possible scope.
"""

from __future__ import annotations

from roverdevkit.surrogate.features import (
    CLASSIFICATION_TARGETS,
    DESIGN_FEATURE_COLUMNS,
    FEASIBILITY_COLUMN,
    INPUT_COLUMNS,
    PRIMARY_REGRESSION_TARGETS,
    REGRESSION_TARGETS,
    SCENARIO_CATEGORICAL_COLUMNS,
    SCENARIO_NUMERIC_COLUMNS,
)


def test_design_feature_count() -> None:
    # Schema v7 (v7 schema follow-up) dropped designed_duty_cycle.
    assert len(DESIGN_FEATURE_COLUMNS) == 11


def test_input_columns_compose_from_groups() -> None:
    expected = DESIGN_FEATURE_COLUMNS + SCENARIO_NUMERIC_COLUMNS + SCENARIO_CATEGORICAL_COLUMNS
    assert expected == INPUT_COLUMNS
    # Schema v7_1 added scenario_operational_duty_cycle; schema v9 added
    # scenario_payload_mass_kg + scenario_payload_power_w to
    # SCENARIO_NUMERIC_COLUMNS so the surrogate sees payload as true inputs.
    assert len(SCENARIO_NUMERIC_COLUMNS) == 12
    assert len(INPUT_COLUMNS) == 27  # 11 + 12 + 4


def test_regression_targets_include_primaries() -> None:
    for col in PRIMARY_REGRESSION_TARGETS:
        assert col in REGRESSION_TARGETS
    assert "range_km" in PRIMARY_REGRESSION_TARGETS
    assert "total_mass_kg" in PRIMARY_REGRESSION_TARGETS


def test_inputs_disjoint_from_targets() -> None:
    assert set(INPUT_COLUMNS).isdisjoint(set(REGRESSION_TARGETS))
    assert set(INPUT_COLUMNS).isdisjoint(set(CLASSIFICATION_TARGETS))


def test_feasibility_classifier_is_stalled_only() -> None:
    """Schema v6 (v6 schema update): the single feasibility classifier is
    ``stalled`` (positive class = infeasible). See ``data/analytical/SCHEMA.md``
    for the v5 -> v6 polarity flip and the v1 -> v2 thermal removal.
    """
    assert CLASSIFICATION_TARGETS == ["stalled"]
    assert FEASIBILITY_COLUMN == "stalled"
    assert "thermal_survival" not in CLASSIFICATION_TARGETS
    assert "thermal_survival" not in REGRESSION_TARGETS
