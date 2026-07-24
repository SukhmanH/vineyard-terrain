"""Unit tests for the phase-5 frost-risk and suitability heuristic.

Assert directions, ordering, and the contract shape, not exact floats. The
parameters are a v1 heuristic slated for tuning against sensor data, so tests
pin behavior (a hollow scores worse than a bench, steep ground is penalized)
rather than magic numbers.
"""
import numpy as np
import pytest

from backend.pipeline import scoring


def _flat_inputs(shape=(50, 50)):
    """Neutral inputs: flat, mid-elevation, no pooling, no drainage."""
    z = np.full(shape, 300.0, dtype="float32")
    tpi = np.zeros(shape, dtype="float32")
    slope_deg = np.zeros(shape, dtype="float32")
    slope_pct = np.zeros(shape, dtype="float32")
    aspect_deg = np.full(shape, 190.0, dtype="float32")  # S-SW optimum
    mask = np.zeros(shape, dtype=bool)
    return z, tpi, slope_deg, slope_pct, aspect_deg, mask


def test_pit_scores_higher_frost_than_bump():
    # A realistic TPI field: a graded slope so a good fraction of cells pool
    # (the p95 normalization needs pooling to be present, as real tiles have).
    z, tpi, sd, sp, asp, mask = _flat_inputs()
    yy, xx = np.mgrid[0:50, 0:50]
    tpi = (-3.0 + 6.0 * (xx / 49.0)).astype("float32")  # west pools, east rises
    frost, suit, stats = scoring.frost_and_suitability(
        z, tpi, sd, sp, asp, None, mask, res=1.0)
    assert np.nanmean(frost[:, :10]) > np.nanmean(frost[:, -10:])


def test_low_ground_raises_frost():
    shape = (50, 50)
    z = np.tile(np.linspace(300, 400, 50, dtype="float32"), (50, 1))  # rises east
    tpi = np.zeros(shape, dtype="float32")
    sd = np.zeros(shape, dtype="float32")
    sp = np.zeros(shape, dtype="float32")
    asp = np.full(shape, 190.0, dtype="float32")
    mask = np.zeros(shape, dtype=bool)
    frost, _, _ = scoring.frost_and_suitability(z, tpi, sd, sp, asp, None, mask, 1.0)
    assert np.nanmean(frost[:, :5]) > np.nanmean(frost[:, -5:])  # low west colder


def test_drainage_component_raises_frost():
    z, tpi, sd, sp, asp, mask = _flat_inputs()
    fa = np.zeros(z.shape, dtype="float32")
    fa[:, 25] = 1.0  # a convergence channel
    frost_with, _, _ = scoring.frost_and_suitability(z, tpi, sd, sp, asp, fa, mask, 1.0)
    frost_without, _, _ = scoring.frost_and_suitability(z, tpi, sd, sp, asp, None, mask, 1.0)
    assert np.nanmean(frost_with[:, 25]) > np.nanmean(frost_without[:, 25])


def test_south_aspect_beats_north_for_suitability():
    z, tpi, sd, sp, asp, mask = _flat_inputs()
    sd = np.full(z.shape, 10.0, dtype="float32")   # sloped so aspect matters
    sp = np.full(z.shape, 10.0, dtype="float32")
    south = asp.copy(); south[:] = 190.0
    north = asp.copy(); north[:] = 10.0
    _, suit_s, _ = scoring.frost_and_suitability(z, tpi, sd, sp, south, None, mask, 1.0)
    _, suit_n, _ = scoring.frost_and_suitability(z, tpi, sd, sp, north, None, mask, 1.0)
    assert np.nanmean(suit_s) > np.nanmean(suit_n)


def test_slope_factor_shape():
    sp = np.array([0.0, 1.9, 2.0, 10.0, 17.0, 23.5, 30.0, 37.5, 45.0, 60.0],
                  dtype="float32")
    f = scoring._slope_factor(sp)
    assert f[0] == pytest.approx(0.85)          # flat
    assert f[3] == pytest.approx(1.0)           # in the sweet spot
    assert f[4] == pytest.approx(1.0)           # 17% still full
    assert 0.6 < f[5] < 1.0                     # 23.5% tapering
    assert f[6] == pytest.approx(0.6)           # 30%
    assert 0.3 < f[7] < 0.6                     # 37.5% tapering
    assert f[8] == pytest.approx(0.3)           # 45%
    assert f[9] == 0.0                          # >45% not mechanizable


def test_unmechanizable_slope_zeros_suitability():
    z, tpi, sd, sp, asp, mask = _flat_inputs()
    sd = np.full(z.shape, 30.0, dtype="float32")
    sp = np.full(z.shape, 60.0, dtype="float32")  # too steep
    _, suit, _ = scoring.frost_and_suitability(z, tpi, sd, sp, asp, None, mask, 1.0)
    assert np.nanmax(suit) == pytest.approx(0.0)


def test_areas_sum_to_valid_area_and_have_acres():
    z, tpi, sd, sp, asp, mask = _flat_inputs((100, 100))
    _, _, stats = scoring.frost_and_suitability(z, tpi, sd, sp, asp, None, mask, 1.0)
    ha = stats["suitability_areas_ha"]
    ac = stats["suitability_areas_acres"]
    total_ha = sum(ha.values())
    assert total_ha == pytest.approx(100 * 100 / 10000.0, abs=0.05)  # 1.0 ha
    for k in ha:
        assert ac[k] == pytest.approx(ha[k] * 2.471, abs=0.05)
    assert set(ha) == {"excellent", "good", "marginal", "avoid"}


def test_masked_cells_are_nan_and_excluded():
    z, tpi, sd, sp, asp, mask = _flat_inputs()
    mask = mask.copy(); mask[:10, :] = True
    frost, suit, _ = scoring.frost_and_suitability(z, tpi, sd, sp, asp, None, mask, 1.0)
    assert np.isnan(frost[:10, :]).all()
    assert np.isnan(suit[:10, :]).all()
    assert np.isfinite(frost[~mask]).all()


def test_percentile_rank_bounds_and_monotonicity():
    v = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    r = scoring._percentile_rank(v)
    assert r.min() == pytest.approx(0.0) and r.max() == pytest.approx(1.0)
    # rank order must follow value order
    assert np.argsort(v).tolist() == np.argsort(r).tolist()
