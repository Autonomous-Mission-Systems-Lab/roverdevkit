"""Consistency gate: the three rover-data consumers must agree with the
canonical published-facts reference (``data/rovers.yaml``).

The canonical file is the single source of truth for *published* /
*derived* facts. This test enforces that every downstream consumer
(mass-validation set, flown-rover truth table, executable registry)
matches those authoritative facts, so the sources cannot silently drift.

``imputed`` facts are model-specific estimates and are NOT enforced.
"""

from __future__ import annotations

import math

from roverdevkit.mass.validation import load_validation_set
from roverdevkit.validation.rover_facts import (
    VALID_PROVENANCE,
    facts_by_name,
    load_rover_facts,
)
from roverdevkit.validation.rover_registry import load_truth_table, registry

# Canonical field -> attribute on the mass-validation row.
_MASS_SET_FIELDS = {
    "mass_total_kg": "mass_total_kg",
    "n_wheels": "n_wheels",
    "wheel_radius_m": "wheel_radius_m",
    "wheel_width_m": "wheel_width_m",
    "grouser_height_m": "grouser_height_m",
    "grouser_count": "grouser_count",
    "solar_area_m2": "solar_area_m2",
    "battery_capacity_wh": "battery_capacity_wh",
    "payload_mass_kg": "payload_mass_kg",
}

# Canonical field -> accessor on a registry entry.
_REGISTRY_FIELDS = {
    "n_wheels": lambda e: e.design.n_wheels,
    "wheel_radius_m": lambda e: e.design.wheel_radius_m,
    "wheel_width_m": lambda e: e.design.wheel_width_m,
    "grouser_height_m": lambda e: e.design.grouser_height_m,
    "grouser_count": lambda e: e.design.grouser_count,
    "wheelbase_m": lambda e: e.design.wheelbase_m,
    "solar_area_m2": lambda e: e.design.solar_area_m2,
    "battery_capacity_wh": lambda e: e.design.battery_capacity_wh,
    "payload_mass_kg": lambda e: getattr(e.scenario, "payload_mass_kg", None),
    "landing_latitude_deg": lambda e: e.scenario.latitude_deg,
}


def _values_match(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------


def test_facts_file_loads_and_is_well_formed() -> None:
    rovers = load_rover_facts()
    assert rovers, "canonical facts file has no rovers"
    seen: set[str] = set()
    for rover in rovers:
        for name in rover.all_names:
            assert name not in seen, f"duplicate rover name/alias {name!r}"
            seen.add(name)
        assert rover.fields, f"{rover.name} has no fields"
        for field_name, fact in rover.fields.items():
            assert fact.provenance in VALID_PROVENANCE, (
                f"{rover.name}.{field_name} has invalid provenance "
                f"{fact.provenance!r}"
            )
            assert fact.source, f"{rover.name}.{field_name} has no source"


# ---------------------------------------------------------------------------
# Consumer consistency
# ---------------------------------------------------------------------------


def test_mass_validation_set_matches_canonical_facts() -> None:
    facts = facts_by_name()
    mismatches: list[str] = []
    for row in load_validation_set():
        rover = facts.get(row.rover_name)
        assert rover is not None, (
            f"mass_validation_set rover {row.rover_name!r} not in data/rovers.yaml"
        )
        for canon_field, attr in _MASS_SET_FIELDS.items():
            fact = rover.fields.get(canon_field)
            if fact is None or not fact.is_enforced:
                continue
            actual = getattr(row, attr)
            if not _values_match(fact.value, actual):
                mismatches.append(
                    f"{rover.name}.{canon_field}: canonical={fact.value!r} "
                    f"mass_set={actual!r}"
                )
    assert not mismatches, "mass_validation_set drifted from canonical facts:\n" + "\n".join(
        mismatches
    )


def test_truth_table_matches_canonical_facts() -> None:
    facts = facts_by_name()
    mismatches: list[str] = []
    for truth in load_truth_table():
        rover = facts.get(truth.rover_name)
        assert rover is not None
        checks = {
            "traverse_m": truth.traverse_m_published,
            "peak_solar_power_w": truth.peak_solar_power_w_published,
            "thermal_survival": truth.thermal_survival_published,
            "mission_duration_days": truth.mission_duration_published_days,
        }
        for canon_field, actual in checks.items():
            fact = rover.truth.get(canon_field)
            if fact is None or not fact.is_enforced:
                continue
            if not _values_match(fact.value, actual):
                mismatches.append(
                    f"{rover.name}.truth.{canon_field}: canonical={fact.value!r} "
                    f"truth_table={actual!r}"
                )
    assert not mismatches, "published_traverse_data drifted from canonical facts:\n" + "\n".join(
        mismatches
    )


def test_registry_matches_canonical_facts() -> None:
    facts = facts_by_name()
    mismatches: list[str] = []
    for entry in registry():
        rover = facts.get(entry.rover_name)
        assert rover is not None, (
            f"registry rover {entry.rover_name!r} not in data/rovers.yaml"
        )
        for canon_field, accessor in _REGISTRY_FIELDS.items():
            fact = rover.fields.get(canon_field)
            if fact is None or not fact.is_enforced:
                continue
            actual = accessor(entry)
            if actual is None:
                continue
            if not _values_match(fact.value, actual):
                mismatches.append(
                    f"{rover.name}.{canon_field}: canonical={fact.value!r} "
                    f"registry={actual!r}"
                )
    assert not mismatches, "rover_registry drifted from canonical facts:\n" + "\n".join(
        mismatches
    )

