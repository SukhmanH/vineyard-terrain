"""Per-parcel metrics (backend/pipeline/zonal.py) on synthetic AOIs.

Routing is supplied as plain grids here (as tests/test_scoring.py does), so these run without
WhiteboxTools; tests/test_context.py runs the real routing.
"""
import numpy as np
import pytest

from backend.pipeline import scoring, zonal

RES = 1.0
CELL_AC = RES * RES / zonal.SQM_PER_ACRE


def _rough(z, seed=0):
    rng = np.random.default_rng(seed)
    return (z + rng.normal(0.0, 0.002, z.shape)).astype("float32")


def _valley(shape=(300, 600)):
    """Flat valley floor (x < 200) at 300 m, a 20 % valley side, then a gentle bench from
    360 m (x >= 500): the same block tests/test_scoring.py uses."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype("float64")
    z = np.where(xx < 200, 300.0 + 0.002 * yy,
                 np.where(xx < 500, 300.0 + 0.2 * (xx - 200),
                          360.0 + 0.01 * (xx - 500)))
    return _rough(z)


def _detail(z, mask=None, routing=None):
    mask = np.zeros(z.shape, bool) if mask is None else mask
    field = scoring.field_surface(z, mask, RES)
    d = scoring.score_detail(z, mask, RES, field=field, routing=routing)
    return d, field["z"], ~mask


def _box(shape, r0, r1, c0, c1):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def test_a_floor_parcel_scores_colder_than_a_bench_parcel_in_the_same_aoi():
    z = _valley()
    d, zs, valid = _detail(z)
    floor = zonal.parcel_metrics(d, zs, _box(z.shape, 100, 200, 50, 150), valid, RES)
    bench = zonal.parcel_metrics(d, zs, _box(z.shape, 100, 200, 520, 590), valid, RES)
    assert floor["frost_score"] < 40 < 70 < bench["frost_score"]
    assert floor["pct_high_risk"] > 90 and bench["pct_high_risk"] == 0.0
    assert floor["drivers_pct"]["valley_floor"] > 90
    assert floor["height_above_floor_median_m"] == pytest.approx(0.0, abs=1.0)
    assert bench["height_above_floor_median_m"] == pytest.approx(61.0, abs=2.0)
    assert bench["frost_safe_acres"] == pytest.approx(bench["plantable_acres"])
    assert floor["frost_safe_acres"] == 0.0
    for m in (floor, bench):
        assert m["label"] == "spring/fall frost risk (terrain)"
        assert m["model_version"] == "v2+z1" and m["coverage_pct"] == 100.0
        assert sum(m["suitability_acres"].values()) == pytest.approx(m["scorable_acres"],
                                                                     abs=0.02)


def test_frost_safe_is_below_40_and_low_risk_below_30():
    """Frost-safe is the owner's < 40 cut; pct_low_risk keeps the < 30 of the spec."""
    assert zonal.FROST_SAFE_MAX == 40.0 and zonal.LOW_RISK_MAX == 30.0
    z = _rough(np.full((200, 200), 400.0) + 0.05 * np.mgrid[0:200, 0:200][1])
    d, zs, valid = _detail(z)
    d["frost"][:, :] = 35.0                       # between the two cuts
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 50, 150, 50, 150), valid, RES)
    assert m["frost_safe_acres"] == pytest.approx(10000 * CELL_AC, abs=0.01)
    assert m["pct_low_risk"] == 0.0 and m["frost_score"] == pytest.approx(65.0)


def test_largest_contiguous_block_is_separate_from_total_safe_acres():
    z = _rough(np.full((200, 200), 400.0) + 0.05 * np.mgrid[0:200, 0:200][1])
    d, zs, valid = _detail(z)
    d["frost"][:, :] = 10.0
    d["frost"][:, 99:101] = 90.0                  # a frost channel splits the parcel
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 0, 200, 0, 200), valid, RES)
    assert m["frost_safe_acres"] == pytest.approx(200 * 198 * CELL_AC, abs=0.01)
    assert m["largest_contiguous_safe_acres"] == pytest.approx(200 * 99 * CELL_AC, abs=0.01)
    d["frost"][100, 100] = 10.0                   # a diagonal-only gap does not join rows
    d["frost"][99, 99] = 90.0
    assert zonal.largest_component(np.eye(3, dtype=bool)) == 1


def test_no_plantable_land_gives_no_score_and_says_so():
    z = _rough(400.0 + 0.6 * np.mgrid[0:200, 0:200][1].astype("float64"))   # 60 % slope
    d, zs, valid = _detail(z)
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 50, 150, 50, 150), valid, RES)
    assert m["frost_score"] is None and "no_plantable_land" in m["flags"]
    assert m["plantable_acres"] == 0.0 and m["frost_risk_mean"] is not None


def test_partial_bare_earth_coverage_is_reported_and_lowers_confidence():
    z = _valley()
    mask = np.zeros(z.shape, bool)
    mask[:, 540:] = True                          # half the bench parcel has no bare earth
    d, zs, valid = _detail(z, mask)
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 100, 200, 520, 560), valid, RES)
    assert m["coverage_pct"] == pytest.approx(50.0)
    assert "low_coverage" in m["flags"] and m["confidence"] == "low"


def test_canopy_is_informational_and_never_reduces_plantable_acres():
    z = _rough(np.full((200, 200), 400.0) + 0.05 * np.mgrid[0:200, 0:200][1])
    d, zs, valid = _detail(z)
    canopy = np.zeros(z.shape, "float32")
    canopy[50:70, 50:100] = 8.0                   # an old orchard block
    inside = _box(z.shape, 50, 150, 50, 150)
    with_trees = zonal.parcel_metrics(d, zs, inside, valid, RES, canopy=canopy)
    bare = zonal.parcel_metrics(d, zs, inside, valid, RES)
    assert with_trees["canopy_over_3m_acres"] == pytest.approx(1000 * CELL_AC, abs=0.01)
    assert with_trees["plantable_acres"] == bare["plantable_acres"]
    assert bare["canopy_over_3m_acres"] is None and "canopy_present" in with_trees["flags"]


def test_water_in_the_parcel_is_excluded_and_reported():
    z = _rough(np.full((200, 200), 400.0) + 0.05 * np.mgrid[0:200, 0:200][1]).astype("float64")
    z[80:120, 80:120] = 401.0                     # hydro-flattened pond
    d, zs, valid = _detail(z.astype("float32"))
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 50, 150, 50, 150), valid, RES)
    assert m["water_acres"] == pytest.approx(1600 * CELL_AC, abs=0.05)
    assert m["scorable_acres"] == pytest.approx((10000 - 1600) * CELL_AC, abs=0.05)
    assert "water_in_parcel" in m["flags"]


def test_drivers_are_measured_inside_the_parcel_only():
    z = _rough(np.full((200, 400), 500.0) + 0.05 * np.mgrid[0:200, 0:400][1])
    routing = {"sca": np.ones(z.shape), "sink_depth": np.zeros(z.shape)}
    routing["sink_depth"][80:120, 280:320] = 2.0  # a closed hollow, east half only
    d, zs, valid = _detail(z, routing=routing)
    east = zonal.parcel_metrics(d, zs, _box(z.shape, 60, 140, 260, 340), valid, RES)
    west = zonal.parcel_metrics(d, zs, _box(z.shape, 60, 140, 60, 140), valid, RES)
    assert east["drivers_pct"]["closed_hollow"] > 90 and east["pct_high_risk"] > 15
    assert west["pct_high_risk"] == 0.0 and sum(west["drivers_pct"].values()) == 0.0


@pytest.mark.parametrize("args, kw, expected", [
    (("high", 100.0), {}, "high"),
    (("high", 90.0), {}, "medium"),
    (("high", 100.0), {"mixed": True, "weak_floor": True}, "low"),
    (("medium", 100.0), {}, "medium"),
    (("medium", 100.0), {"buffer_inadequate": True}, "low"),
    (("high", 79.9), {}, "low"),
    (("low", 100.0), {}, "low"),
])
def test_confidence_ladder(args, kw, expected):
    assert zonal.confidence_level(*args, **kw) == expected


def test_flags_follow_the_context():
    z = _valley()
    d, zs, valid = _detail(z)
    m = zonal.parcel_metrics(d, zs, _box(z.shape, 100, 200, 520, 590), valid, RES,
                             edge_inflow_pct=12.0, nodata_inflow_pct=7.0, widened=True,
                             mix={"refused": True}, aoi_resampled_m=2.0,
                             parcel_flags=["multi_pid"])
    for f in ("multi_pid", "buffer_widened", "buffer_inadequate", "upslope_nodata",
              "delivery_mix_refused", "aoi_resampled_2m", "floor_from_block"):
        assert f in m["flags"], f
    assert m["confidence"] == "medium"            # high, less one for the short buffer


# ------------------------------------------------------------- inflow edges
def test_inflow_cells_are_on_the_rim_the_terrain_rises_toward():
    yy, xx = np.mgrid[0:100, 0:80].astype("float64")
    zs = 400.0 - 0.1 * yy                         # rises to the north (row 0)
    valid, water = np.ones(zs.shape, bool), np.zeros(zs.shape, bool)
    rim, hole = zonal.inflow_cells(zs, valid, water, 1.0)
    assert rim[0].all() and not rim[1:].any() and not hole.any()
    valid[30:40, 20:30] = False                   # a NoData gap: its uphill (north) edge
    rim, hole = zonal.inflow_cells(np.where(valid, zs, np.nan), valid, water, 1.0)
    assert hole[40, 20:30].all() and not hole[29, 20:30].any()
    water[60:70, 20:30], valid[60:70, 20:30] = True, True
    rim2, hole2 = zonal.inflow_cells(zs, np.ones(zs.shape, bool), water, 1.0)
    assert not hole2.any()                        # lakes are outlets, not inflow


def test_reached_pct_counts_cells_with_more_than_half_a_cell_of_inflow():
    flux = np.array([[0.0, 0.4, 0.6, 3.0]])
    sel = np.ones(flux.shape, bool)
    assert zonal.reached_pct(flux, sel) == 50.0
    assert zonal.reached_pct(None, sel) is None
