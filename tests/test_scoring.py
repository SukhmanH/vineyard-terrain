"""Unit tests for the phase-5 frost-risk and suitability model (v2).

Assert directions, ordering, and the contract shape, not exact floats. The
constants are slated for tuning against sensor data, so tests pin physical
behavior (a valley floor scores colder than the bench above it, a closed
hollow colder than open ground, a south slope warmer than a north slope,
every block scored on one fixed scale rather than against itself).
"""
import numpy as np
import pytest
from rasterio.transform import from_origin

from backend.pipeline import scoring

RES = 1.0


def _rough(z, seed=0):
    """Millimetre noise. Real LiDAR ground is never exactly level, and an
    exactly level synthetic plane would be classed as hydro-flattened water."""
    rng = np.random.default_rng(seed)
    return (z + rng.normal(0.0, 0.002, z.shape)).astype("float32")


def _plane(shape=(300, 300), dz_dx=0.0, dz_dy=0.0, base=300.0):
    """Tilted plane. dz_dx rises toward +x (east), dz_dy toward +y (south in
    array-row order), so dz_dy < 0 faces south."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype("float64")
    return _rough(base + dz_dx * xx + dz_dy * yy)


def _gauss(shape, amp, sigma, cy, cx):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype("float64")
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _valley(shape=(300, 600)):
    """Flat valley floor (x < 200) at 300 m, a 20 pct valley side, then a
    gently sloping bench from 360 m (x >= 500)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype("float64")
    z = np.where(xx < 200, 300.0 + 0.002 * yy,
                 np.where(xx < 500, 300.0 + 0.2 * (xx - 200),
                          360.0 + 0.01 * (xx - 500)))
    return _rough(z)


def _no_mask(z):
    return np.zeros(z.shape, dtype=bool)


def _routing(shape, sca=1.0, sink=0.0):
    return {"sca": np.full(shape, sca, dtype="float64"),
            "sink_depth": np.full(shape, sink, dtype="float64")}


# ------------------------------------------------------------- frost physics
def test_valley_floor_scores_colder_than_bench():
    z = _valley()
    frost, _, st = scoring.frost_and_suitability(z, _no_mask(z), RES)
    assert np.nanmean(frost[:, 20:180]) > 60          # floor: cold pool
    assert np.nanmean(frost[:, 520:580]) < 30         # bench 60 m up: thermal belt
    assert st["frost"]["floor_source"] == "block"
    assert st["frost"]["floor_confidence"] > 0.9
    assert st["frost"]["floor_m"] == pytest.approx(300.0, abs=1.0)
    assert st["frost"]["drivers_pct"]["valley_floor"] > 50


def test_blocks_are_scored_on_a_fixed_scale():
    """v1 ranked each block against itself, so every block came out about
    8 pct high risk, valley floor or bench. A uniform slope has nowhere for
    cold air to collect and must score no high-risk ground at all."""
    valley = _valley()
    slope = _plane(dz_dx=0.15)
    _, _, st_valley = scoring.frost_and_suitability(valley, _no_mask(valley), RES)
    frost_s, _, st_slope = scoring.frost_and_suitability(slope, _no_mask(slope), RES)
    assert st_valley["frost"]["pct_high_risk"] > 25
    assert st_slope["frost"]["pct_high_risk"] == 0.0
    assert np.nanmax(frost_s) < 30
    assert st_slope["frost"]["floor_confidence"] < 0.1   # lowest ground is a slope


def test_valley_reference_keeps_a_bench_block_out_of_the_cold_pool():
    """A bench-only block cannot see the floor below it; its own lowest flat
    ground looks like a floor. The regional valley floor fixes that."""
    yy, xx = np.mgrid[0:300, 0:600].astype("float64")
    bench = _rough(np.where(xx < 300, 420.0 + 0.002 * yy,
                            420.0 + 0.2 * (xx - 300)))   # flat bench + hillside
    m = _no_mask(bench)
    own, _, st_own = scoring.frost_and_suitability(bench, m, RES)
    ref, _, st_ref = scoring.frost_and_suitability(
        bench, m, RES, floor_rows=np.full(300, 342.0))   # lake 78 m below
    assert st_own["frost"]["floor_source"] == "block"
    assert st_ref["frost"]["floor_source"] == "valley"
    assert st_ref["frost"]["floor_m"] == pytest.approx(342.0)
    assert np.nanmean(ref[:, 20:280]) < np.nanmean(own[:, 20:280]) - 30
    assert np.nanmean(ref[:, 20:280]) < 20


def test_valley_reference_drops_to_meet_lower_block_ground():
    z = _valley()   # floor at 300 m
    _, _, st = scoring.frost_and_suitability(
        z, _no_mask(z), RES, floor_rows=np.full(z.shape[0], 305.0))
    assert st["frost"]["floor_m"] == pytest.approx(300.0, abs=1.0)


def test_closed_hollow_raises_frost():
    z = _plane(dz_dx=0.05)
    r = _routing(z.shape)
    r["sink_depth"][100:140, 100:140] = 2.0
    frost, _, st = scoring.frost_and_suitability(z, _no_mask(z), RES, routing=r)
    assert np.nanmean(frost[110:130, 110:130]) > 60
    assert np.nanmean(frost[200:260, 200:260]) < 20
    assert st["frost"]["drivers_pct"]["closed_hollow"] > 90


def test_cold_air_converging_on_gentle_ground_pools_but_drains_off_steep():
    gentle, steep = _plane(dz_dx=0.02), _plane(dz_dx=0.30)
    r = _routing(gentle.shape)
    r["sca"][:, 150] = 2000.0     # a drainage line carrying 2000 m of slope
    fg, _, _ = scoring.frost_and_suitability(gentle, _no_mask(gentle), RES, routing=r)
    fs, _, _ = scoring.frost_and_suitability(steep, _no_mask(steep), RES, routing=r)
    channel = (slice(50, 250), slice(148, 153))
    assert np.nanmean(fg[channel]) > np.nanmean(fs[channel]) + 15
    assert np.nanmean(fg[channel]) > np.nanmean(fg[50:250, 50:100]) + 15


def test_pit_scores_higher_frost_than_bump():
    base = _plane(dz_dx=0.03).astype("float64")
    pit = (base - _gauss(base.shape, 6.0, 25.0, 150, 150)).astype("float32")
    bump = (base + _gauss(base.shape, 6.0, 25.0, 150, 150)).astype("float32")
    fp, _, _ = scoring.frost_and_suitability(pit, _no_mask(pit), RES)
    fb, _, _ = scoring.frost_and_suitability(bump, _no_mask(bump), RES)
    assert np.nanmean(fp[140:160, 140:160]) > np.nanmean(fb[140:160, 140:160]) + 30


def test_plane_tpi_has_no_edge_or_gap_bias():
    """A plain window mean makes the downhill edge of a uniform slope (and
    the rim of a data gap) read as a hollow. The fitted plane does not."""
    z = _plane((300, 400), dz_dx=0.2).astype("float64")
    valid = np.ones(z.shape, dtype=bool)
    valid[100:200, 250:] = False
    t = scoring.plane_tpi(z, valid, 201)
    assert np.abs(t[valid]).max() < 0.05
    pit = z - _gauss(z.shape, 6.0, 20.0, 60, 150)
    assert scoring.plane_tpi(pit, valid, 201)[60, 150] < -4.0


def test_water_is_excluded_from_scores_and_areas():
    z = _plane(dz_dx=0.05).astype("float64")
    z[100:160, 100:160] = 301.0           # hydro-flattened pond: exactly level
    z = z.astype("float32")
    frost, suit, st = scoring.frost_and_suitability(z, _no_mask(z), RES)
    assert np.isnan(frost[110:150, 110:150]).all()
    assert np.isnan(suit[110:150, 110:150]).all()
    assert st["water_ha"] == pytest.approx(0.36, abs=0.05)   # 60 x 60 m pond
    # areas are reported to 0.1 ha, so five rounded terms can drift 0.25
    total = sum(st["suitability_areas_ha"].values()) + st["water_ha"]
    assert total == pytest.approx(z.size * RES * RES / 1e4, abs=0.25)


# ------------------------------------------------------------- heat physics
def _sun(slope_deg, aspect_deg, lat=49.2):
    return float(scoring.sun_ratio(np.array([[slope_deg]]),
                                   np.array([[aspect_deg]]), lat)[0, 0])


def test_sun_ratio_follows_slope_and_aspect():
    assert _sun(0, 0) == pytest.approx(1.0, abs=1e-6)
    assert _sun(10, 190) > 1.03 > 0.97 > _sun(10, 0)       # south warm, north cool
    assert _sun(25, 0) < _sun(10, 0) - 0.1                  # steeper north, cooler
    assert _sun(10, 270) > _sun(10, 90)                     # warm afternoon sun
    best = max(range(0, 360, 2), key=lambda a: _sun(10, a))
    assert 180 <= best <= 225                               # S to SW optimum


def test_sun_ratio_has_no_jump_at_low_slope():
    """v1 switched aspect on at 2 deg, so a 2.1 deg north slope scored far
    below a 1.9 deg one. Sun varies smoothly with slope."""
    assert abs(_sun(1.9, 0) - _sun(2.1, 0)) < 0.01
    assert _sun(2, 0) > 0.97          # a gentle north slope is barely cooler


def test_south_aspect_beats_north_for_suitability():
    south = _plane(dz_dy=-0.10)
    north = _plane(dz_dy=0.10)
    _, suit_s, _ = scoring.frost_and_suitability(south, _no_mask(south), RES)
    _, suit_n, _ = scoring.frost_and_suitability(north, _no_mask(north), RES)
    assert np.nanmean(suit_s) > np.nanmean(suit_n) + 5


# ------------------------------------------------------------- slope + contract
def test_slope_factor_shape():
    sp = np.array([0.0, 2.0, 3.0, 10.0, 17.0, 23.5, 30.0, 37.5, 45.0, 60.0])
    f = scoring._slope_factor(sp)
    assert f[0] == pytest.approx(0.85)          # dead flat
    assert 0.85 < f[1] < 1.0                    # easing in, no step
    assert f[2] == pytest.approx(1.0)
    assert f[3] == pytest.approx(1.0)           # in the sweet spot
    assert f[4] == pytest.approx(1.0)           # 17% still full
    assert 0.6 < f[5] < 1.0                     # 23.5% tapering
    assert f[6] == pytest.approx(0.6)           # 30%
    assert 0.3 < f[7] < 0.6                     # 37.5% tapering
    assert f[8] == pytest.approx(0.3)           # 45%
    assert f[9] == 0.0                          # >45% not mechanizable


def test_unmechanizable_slope_zeros_suitability():
    z = _plane(dz_dx=0.60)
    _, suit, _ = scoring.frost_and_suitability(z, _no_mask(z), RES)
    assert np.nanmax(suit[20:-20, 20:-20]) == pytest.approx(0.0)


def test_areas_sum_to_valid_area_and_have_acres():
    z = _valley()
    _, _, st = scoring.frost_and_suitability(z, _no_mask(z), RES)
    ha, ac = st["suitability_areas_ha"], st["suitability_areas_acres"]
    assert set(ha) == {"excellent", "good", "marginal", "avoid"}
    # each class is reported to 0.1 ha, so four rounded terms can drift 0.2
    assert sum(ha.values()) == pytest.approx(z.size / 1e4, abs=0.2)
    for k in ha:
        assert ac[k] == pytest.approx(ha[k] * 2.471, abs=0.05)
    assert sum(st["frost"]["drivers_pct"].values()) == pytest.approx(100, abs=0.5)


def test_masked_cells_are_nan_and_excluded():
    z = _valley()
    mask = _no_mask(z)
    mask[:30, :] = True
    frost, suit, _ = scoring.frost_and_suitability(z, mask, RES)
    assert np.isnan(frost[:30, :]).all() and np.isnan(suit[:30, :]).all()
    assert np.isfinite(frost[~mask]).all() and np.isfinite(suit[~mask]).all()


def test_all_nodata_returns_empty_stats():
    z = _plane()
    frost, suit, st = scoring.frost_and_suitability(z, np.ones(z.shape, bool), RES)
    assert np.isnan(frost).all() and np.isnan(suit).all()
    assert st["frost"]["pct_high_risk"] == 0.0
    assert sum(st["suitability_areas_ha"].values()) == 0.0


# ------------------------------------------------------------- geography
def test_valley_floor_profile_covers_the_south_okanagan_and_lower_similkameen():
    lats = np.array([49.5063, 49.2063, 49.0938, 49.205, 49.175, 49.020, 49.05, 49.0775])
    lons = np.array([-119.5625, -119.5375, -119.5375, -119.835, -119.76, -119.72, -119.35,
                     -119.6325])
    floor, dist = scoring.valley_floor_profile(lats, lons)
    assert floor[0] == pytest.approx(342.0)                # Penticton bench: lake
    assert floor[1] == pytest.approx(296.5, abs=2.0)       # Oliver
    assert floor[2] == pytest.approx(280.0, abs=2.0)       # north Osoyoos
    assert floor[3] == pytest.approx(409.0, abs=0.5)       # Keremeos
    assert floor[4] == pytest.approx(392.0, abs=0.5)       # Cawston
    assert floor[5] == pytest.approx(359.5, abs=0.5)       # Chopaka
    assert np.isnan(floor[6]) and dist[6] > 4              # Anarchist Mountain: off axis
    # the two lines are not joined: the ridge between Osoyoos and Keremeos is off both
    assert np.isnan(floor[7]) and dist[7] > 4


def test_block_geography_reads_row_floors_from_the_tile_crs():
    meta = {"crs": "EPSG:2955", "transform": from_origin(312000.0, 5452000.0, 1.0, 1.0),
            "width": 1900, "height": 1480}          # Oliver, NAD83(CSRS) UTM 11N
    geo = scoring.block_geography(meta)
    assert 49.15 < geo["lat"] < 49.25
    rows = geo["floor_rows"]
    assert rows is not None and rows.shape == (1480,)
    assert rows[-1] < rows[0]                        # floor falls toward Osoyoos
    far = dict(meta, transform=from_origin(250000.0, 5452000.0, 1.0, 1.0))
    assert scoring.block_geography(far)["floor_rows"] is None
