"""Loader for the canonical published-facts reference (``data/rovers.yaml``).

This module is the single read path for the published, citable facts about
the rovers used in verification. It is intentionally generic: each rover is a
``RoverFacts`` with a name, aliases, and a flat mapping of field name to a
:class:`Provenanced` value (plus an optional ``truth`` block for flown rovers).

Why generic rather than a fixed dataclass schema? The canonical file holds
*published facts only* and is meant to grow (new rovers, new cited fields)
without a code change. The consistency gate in ``tests/test_rover_facts.py``
reads the same generic structure to check that the mass-validation set, the
flown-rover truth table, and the executable registry agree with the
``published`` / ``derived`` facts recorded here.

See ``data/rovers.yaml`` for the facts-vs-modeling split and the per-field
provenance convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FACTS_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "rovers.yaml"

#: Provenance values whose recorded value is authoritative and therefore
#: enforced against the downstream consumers. ``imputed`` values are
#: model-specific estimates and are not enforced.
ENFORCED_PROVENANCE: frozenset[str] = frozenset({"published", "derived"})

VALID_PROVENANCE: frozenset[str] = frozenset({"published", "derived", "imputed"})


@dataclass(frozen=True)
class Provenanced:
    """A single fact: its value plus where the value came from.

    ``low`` / ``high`` are populated only for truth-band fields (e.g. the
    published traverse / peak-solar bands); they are ``None`` otherwise.
    """

    value: Any
    provenance: str
    source: str
    low: float | None = None
    high: float | None = None

    @property
    def is_enforced(self) -> bool:
        """True iff this fact is authoritative (consumers must match it)."""
        return self.provenance in ENFORCED_PROVENANCE


@dataclass(frozen=True)
class RoverFacts:
    """Published facts for one rover."""

    name: str
    aliases: tuple[str, ...]
    fields: dict[str, Provenanced]
    truth: dict[str, Provenanced]

    @property
    def all_names(self) -> tuple[str, ...]:
        """Canonical name plus any aliases (for matching consumer keys)."""
        return (self.name, *self.aliases)


def _parse_provenanced(field_name: str, raw: dict[str, Any]) -> Provenanced:
    if "value" not in raw:
        raise ValueError(f"field {field_name!r} is missing a 'value' key")
    provenance = raw.get("provenance")
    if provenance not in VALID_PROVENANCE:
        raise ValueError(
            f"field {field_name!r} has invalid provenance {provenance!r}; "
            f"must be one of {sorted(VALID_PROVENANCE)}"
        )
    return Provenanced(
        value=raw["value"],
        provenance=provenance,
        source=raw.get("source", ""),
        low=raw.get("low"),
        high=raw.get("high"),
    )


def load_rover_facts(path: Path | str | None = None) -> list[RoverFacts]:
    """Read ``data/rovers.yaml`` into a list of :class:`RoverFacts`."""
    facts_path = Path(path) if path else DEFAULT_FACTS_PATH
    with facts_path.open() as fh:
        doc = yaml.safe_load(fh)

    rovers: list[RoverFacts] = []
    for entry in doc["rovers"]:
        name = entry["name"]
        aliases = tuple(entry.get("aliases", []) or [])
        fields: dict[str, Provenanced] = {}
        truth: dict[str, Provenanced] = {}
        for key, raw in entry.items():
            if key in ("name", "aliases"):
                continue
            if key == "truth":
                for tkey, traw in raw.items():
                    truth[tkey] = _parse_provenanced(f"{name}.truth.{tkey}", traw)
                continue
            fields[key] = _parse_provenanced(f"{name}.{key}", raw)
        rovers.append(RoverFacts(name=name, aliases=aliases, fields=fields, truth=truth))
    return rovers


def facts_by_name(path: Path | str | None = None) -> dict[str, RoverFacts]:
    """Map every canonical name *and* alias to its :class:`RoverFacts`."""
    out: dict[str, RoverFacts] = {}
    for rover in load_rover_facts(path):
        for name in rover.all_names:
            out[name] = rover
    return out
