"""Bottom-up parametric mass model for lunar micro-rovers.

See :mod:`.parametric_mers` for the :func:`estimate_mass` top-level
function and the :class:`MassModelParams` constants bag. See
:mod:`.validation` for the published-rover cross-check. The design choice
to go bottom-up (instead of fitting per-subsystem MERs on n~8 published
rovers) is documented inline in the validation helpers.
"""

from roverdevkit.mass.parametric_mers import (
    MassBreakdown,
    MassModelParams,
    estimate_mass,
    estimate_mass_from_design,
)
from roverdevkit.mass.validation import (
    RoverValidationResult,
    RoverValidationRow,
    ValidationSummary,
    format_report,
    load_validation_set,
    predict_row,
    validate_against_published_rovers,
)

__all__ = [
    "MassBreakdown",
    "MassModelParams",
    "RoverValidationResult",
    "RoverValidationRow",
    "ValidationSummary",
    "estimate_mass",
    "estimate_mass_from_design",
    "format_report",
    "load_validation_set",
    "predict_row",
    "validate_against_published_rovers",
]
