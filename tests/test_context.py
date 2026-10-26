"""Clip last, not first: the parcel path analyzes parcel + buffer, then summarizes.

Each test analyzes the same synthetic ground twice, once in context (the AOI) and once
clipped to the parcel first, as the manual upload path's boundary clip does, and shows what
clipping first gets wrong. These run real WhiteboxTools routing and skip without it.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from backend.pipeline import hydrology, run, tiles, zonal
from tests.helpers import X0, X_OFF_AXIS, Y0, cells_box, entry_for, grid_meta, rough, \
    write_array


@pytest.fixture(scope="module", autouse=True)
def _whitebox():
    z = rough(300.0 + 0.05 * np.mgrid[0:40, 0:40][0].astype("float64"))
    if hydrology.cold_air_routing(z, np.zeros(z.shape, bool), grid_meta(X0, Y0, 40, 40),
                                  tempfile.mkdtemp()) is None:
        pytest.skip("WhiteboxTools unavailable in this environment")


def _metrics(z, mask, meta, parcel):
    a = run.analyze_in_context(z, mask, meta, parcel, tempfile.mkdtemp())
    return zonal.parcel_metrics(a["detail"], a["field_z"], a["inside"], ~mask, 1.0), a


def _both(z, meta, x0, cells):
    """(in context, clipped first) metrics for a parcel given as grid cells."""
    parcel = cells_box(x0, Y0, *cells)
    in_context, _ = _metrics(z, np.zeros(z.shape, bool), meta, parcel)
    clip = np.ones(z.shape, bool)
    r0, r1, c0, c1 = cells
    clip[r0:r1, c0:c1] = False
    clipped, _ = _metrics(np.where(clip, np.nan, z).astype("float32"), clip, meta, parcel)
    return in_context, clipped


def _toe_slope(ny=1100, nx=500):
    """A crest at row 200, a 25 % slope falling south to row 900, then a 1 % toe."""
    row = np.mgrid[0:ny, 0:nx][0].astype("float64")
    return rough(np.where(row < 200, 875.0 - 0.1 * (200 - row),
                          np.where(row <= 900, 700.0 + 0.25 * (900 - row),
                                   700.0 - 0.01 * (row - 900))))


def test_cold_air_draining_from_upslope_reaches_a_toe_parcel_only_in_context():
    z = _toe_slope()
    ctx, clipped = _both(z, grid_meta(X0, Y0, *z.shape), X0, (920, 1020, 200, 300))
    assert ctx["drivers_pct"]["cold_air_flow"] > 90 and ctx["pct_high_risk"] > 15
    assert clipped["pct_high_risk"] == 0.0          # the upslope catchment was deleted
    assert ctx["frost_risk_mean"] > clipped["frost_risk_mean"] + 20


def test_an_embankment_past_the_property_line_dams_cold_air_only_in_context():
    ny, nx = 600, 400
    r, c = np.mgrid[0:ny, 0:nx].astype("float64")
    swale = 700.0 - 0.03 * r + 0.05 * np.abs(c - 200)
    berm = (r >= 410) & (r < 420) & (np.abs(c - 200) <= 100)   # 10 m south of the parcel
    z = rough(swale + np.where(berm, 2.5, 0.0))
    ctx, clipped = _both(z, grid_meta(X0, Y0, ny, nx), X0, (300, 400, 150, 250))
    assert ctx["drivers_pct"]["closed_hollow"] > 30
    assert clipped["drivers_pct"]["closed_hollow"] == 0.0     # the dam was cut off
    assert ctx["frost_risk_mean"] > clipped["frost_risk_mean"] + 20


def _off_axis_valley(ny=300, nx=800):
    """Off the valley-axis table: a 300 m floor (cols < 300), a 20 % side, a 1 % bench."""
    r, c = np.mgrid[0:ny, 0:nx].astype("float64")
    return rough(np.where(c < 300, 300.0 + 0.002 * r,
                          np.where(c < 500, 300.0 + 0.2 * (c - 300),
                                   340.0 + 0.01 * (c - 500))))


def test_a_valley_floor_parcel_is_only_recognised_as_floor_in_context():
    """Clipped to itself, a floor parcel has no valley sides, so its floor confidence is
    ~0 and it scores as if safe: the most expensive mistake this tool can make."""
    z = _off_axis_valley()
    ctx, clipped = _both(z, grid_meta(X_OFF_AXIS, Y0, *z.shape), X_OFF_AXIS,
                         (100, 200, 100, 200))
    assert ctx["floor_source"] == "block" and ctx["floor_confidence"] > 0.9
    assert ctx["frost_risk_mean"] > 70 and ctx["drivers_pct"]["valley_floor"] > 90
    assert clipped["floor_confidence"] < 0.1
    assert clipped["frost_risk_mean"] < ctx["frost_risk_mean"] - 40


def test_the_parcels_own_lowest_corner_is_not_taken_as_the_valley_floor():
    z = _off_axis_valley()
    ctx, clipped = _both(z, grid_meta(X_OFF_AXIS, Y0, *z.shape), X_OFF_AXIS,
                         (100, 200, 600, 700))
    assert ctx["floor_m"] == pytest.approx(300.0, abs=1.0)
    assert clipped["floor_m"] == pytest.approx(341.0, abs=1.0)    # its own lowest ground
    assert ctx["height_above_floor_median_m"] == pytest.approx(41.5, abs=2.0)


# ------------------------------------------------------------- run_parcel
def _resolution(geom, **kw):
    from backend.pipeline import parcel
    res = {"status": "resolved", "pid": "000-000-001", "pids": ["000-000-001"],
           "geometry_3005": geom, "geometry_wgs84": parcel.to_wgs84(geom),
           "area_acres": round(geom.area / 4046.86, 2), "owner_type": "Private",
           "parcel_class": "Subdivision", "source": "pid", "confidence": "high",
           "candidates": [], "reasons": [], "flags": [], "point_3005": None}
    res.update(kw)
    return res


@pytest.fixture
def toe_tiles(tmp_path, monkeypatch):
    """The toe-slope ground as two 2024 tiles (north and south halves) in a fake index."""
    z = _toe_slope()
    north = write_array(str(tmp_path / "n.tif"), z[:550], X0, Y0)
    south = write_array(str(tmp_path / "s.tif"), z[550:], X0, Y0 - 550)
    entries = [entry_for(north, 2024, "bc_082e023_1_2_1_xli1m_utm11_x.tif"),
               entry_for(south, 2024, "bc_082e023_1_2_3_xli1m_utm11_x.tif")]
    local = {entries[0]["key"]: north, entries[1]["key"]: south}
    monkeypatch.setattr(tiles, "fetch", lambda e, window=None: local[e["key"]])
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path / "data"))
    return entries


def test_run_parcel_writes_the_contract_and_the_parcel_block(tmp_path, toe_tiles):
    out = tmp_path / "job"
    seen = []
    geom = cells_box(X0, Y0, 920, 1020, 200, 300)
    stats, prov = run.run_parcel(_resolution(geom), str(out), buffer_m=800,
                                 entries=toe_tiles, on_status=seen.append, label="Toe")
    assert seen == ["fetching", "analyzing"]
    for name in ("stats.json", "meta.json", "bounds.json", "parcel.geojson"):
        assert (out / name).is_file(), name
    for layer in ("elevation", "slope", "tpi", "flowacc", "frost", "suitability"):
        assert (out / "layers" / f"{layer}.png").is_file(), layer
    assert not (out / "_hydro").exists()

    saved = json.loads((out / "stats.json").read_text())
    p = saved["parcel"]
    assert p == json.loads(json.dumps(stats["parcel"]))
    assert p["pid"] == "000-000-001" and p["model_version"] == "v2+z1"
    assert p["coverage_pct"] == 100.0 and p["edge_inflow_pct"] == 0.0
    assert p["drivers_pct"]["cold_air_flow"] > 90 and p["confidence"] == "high"
    assert saved["flags"]["is_dsm"] is False and saved["flags"]["canopy_caveat"] is False
    assert "frost" in saved and "coverage" in saved            # AOI-level stats too

    meta = json.loads((out / "meta.json").read_text())
    assert (meta["kind"], meta["product"], meta["filename"]) == ("parcel", "dem", "Toe")
    assert meta["is_dsm"] is False and meta["provenance"]["years"] == [2024]
    assert sorted(meta["provenance"]["tile_keys"]) == sorted(e["key"] for e in toe_tiles)
    assert prov["buffer_m"] == 800 and prov["buffer_widened"] is False
    assert prov["model_version"] == "v2+z1" and prov["git_sha"]

    fc = json.loads((out / "parcel.geojson").read_text())
    assert [f["properties"]["role"] for f in fc["features"]] == ["parcel", "aoi"]


def test_cold_air_across_the_aoi_edge_widens_once_then_flags(tmp_path, toe_tiles):
    """A 100 m buffer cuts a 700 m slope that drains straight onto the parcel."""
    geom = cells_box(X0, Y0, 920, 1020, 200, 300)
    stats, prov = run.run_parcel(_resolution(geom), str(tmp_path / "job"), buffer_m=100,
                                 entries=toe_tiles)
    p = stats["parcel"]
    assert prov["buffer_widened"] is True and prov["buffer_m"] == 200
    assert p["edge_inflow_pct"] > zonal.EDGE_INFLOW_MAX_PCT
    assert "buffer_widened" in p["flags"] and "buffer_inadequate" in p["flags"]
    assert p["confidence"] == "medium"


def test_a_routing_failure_fails_the_job_instead_of_scoring_without_it(tmp_path, toe_tiles,
                                                                        monkeypatch):
    monkeypatch.setattr(hydrology, "cold_air_routing", lambda *a, **k: None)
    geom = cells_box(X0, Y0, 920, 1020, 200, 300)
    with pytest.raises(run.ParcelAnalysisError, match="not be comparable"):
        run.run_parcel(_resolution(geom), str(tmp_path / "job"), buffer_m=300,
                       entries=toe_tiles)
    assert not os.path.exists(tmp_path / "job" / "_hydro")


def test_an_unresolved_parcel_is_never_analyzed(tmp_path):
    with pytest.raises(run.ParcelAnalysisError, match="not resolved"):
        run.run_parcel({"status": "needs_confirmation"}, str(tmp_path / "job"))


def test_off_axis_parcels_get_a_wider_buffer():
    on_axis = cells_box(X0, Y0, 0, 100, 0, 100)
    off_axis = cells_box(X_OFF_AXIS, Y0, 0, 100, 0, 100)
    assert run.choose_buffer(on_axis, 750) == (750.0, False)
    assert run.choose_buffer(off_axis, 750) == (run.OFF_AXIS_BUFFER_M, True)
    assert run.choose_buffer(off_axis, 1500) == (1500.0, True)
