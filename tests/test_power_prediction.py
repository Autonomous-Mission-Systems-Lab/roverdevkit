"""Tests for the §5.3 de-tuned (no per-rover calibration) peak-solar prediction.

The point of the de-tuned predictor is that it must *not* use each rover's
registry ``panel_efficiency`` / ``panel_dust_factor`` (which were tuned to that
rover's published number). These tests pin:

1. The fixed literature parameter stack-up and its uniform application.
2. That the prediction depends only on published geometry, not the registry's
   tuned per-rover panel knobs.
3. The headline honest result: the fresh-array rover (Pragyan) lands in-band
   while the multi-year rover (Yutu-2) over-predicts and exposes a degradation
   derate well below 1.
"""

from __future__ import annotations

from roverdevkit.power.solar import (
    SOLAR_CONSTANT_AU_1_W_PER_M2,
    panel_power_w,
    sun_elevation_deg,
)
from roverdevkit.validation.power_prediction import (
    CELL_EFFICIENCY_BOL,
    CLEAN_DUST_FACTOR,
    ELECTRICAL_DERATE,
    HIGH_TEMP_DERATE,
    PACKING_FACTOR,
    SYSTEM_EFFICIENCY,
    predict_all_flown,
    sensitivity_band_w,
)


def _by_name() -> dict[str, object]:
    return {p.rover_name: p for p in predict_all_flown()}


def test_system_efficiency_is_the_cited_product() -> None:
    assert SYSTEM_EFFICIENCY == (
        CELL_EFFICIENCY_BOL * PACKING_FACTOR * ELECTRICAL_DERATE * HIGH_TEMP_DERATE
    )
    # Sanity: net system efficiency sits below the bare cell BOL value.
    assert 0.18 < SYSTEM_EFFICIENCY < CELL_EFFICIENCY_BOL


def test_prediction_ignores_registry_tuned_panel_params() -> None:
    # The de-tuned clean prediction must equal a forward panel_power_w call
    # using the *uniform* literature SYSTEM_EFFICIENCY -- not the registry's
    # per-rover panel_efficiency (Pragyan 0.22, Yutu-2 0.20).
    preds = _by_name()
    for p in preds.values():
        peak_elev = sun_elevation_deg(p.latitude_deg, lunar_hour_angle_deg=0.0)
        expected = panel_power_w(
            panel_area_m2=p.panel_area_m2,
            panel_efficiency=SYSTEM_EFFICIENCY,
            sun_elevation_deg=peak_elev,
            panel_tilt_deg=0.0,
            dust_degradation_factor=CLEAN_DUST_FACTOR,
            solar_constant_w_per_m2=SOLAR_CONSTANT_AU_1_W_PER_M2,
        )
        assert abs(p.predicted_clean_w - expected) < 1e-6


def test_sensitivity_band_brackets_the_clean_prediction() -> None:
    for p in _by_name().values():
        lo, hi = sensitivity_band_w(p.panel_area_m2, p.peak_elevation_deg)
        assert lo <= p.predicted_clean_w <= hi
        assert p.sensitivity_low_w == lo
        assert p.sensitivity_high_w == hi


def test_fresh_array_predicts_in_band() -> None:
    pragyan = _by_name()["Pragyan"]
    assert pragyan.in_band
    assert abs(pragyan.pct_error_vs_published) < 15.0
    # Near-fresh array: implied derate close to 1.
    assert pragyan.implied_total_derate > 0.8


def test_aged_array_over_predicts_and_exposes_derate() -> None:
    yutu = _by_name()["Yutu-2"]
    assert not yutu.in_band
    assert yutu.predicted_bol_w > yutu.band_high_w
    # Multi-year dust + EOL: published value implies a large degradation.
    assert yutu.implied_total_derate < 0.65
    # The fresh rover should be far less degraded than the aged one.
    assert yutu.implied_total_derate < _by_name()["Pragyan"].implied_total_derate
