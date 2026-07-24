"""Phase-6 tests: multi-tile mosaic (incl. mixed CRS) and parcel clipping."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.pipeline import loader, run

NODATA = -32767.0


def _write_tile(path, origin_x, origin_y, crs="EPSG:2955", ny=120, nx=120,
                base=300.0, res=1.0):
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    z = base + 0.03 * xx + 0.01 * yy
    transform = from_origin(origin_x, origin_y, res, res)
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="float32", crs=crs, transform=transform,
                       nodata=NODATA) as ds:
        ds.write(z.astype("float32"), 1)


def test_single_path_matches_load_dem(tmp_path):
    p = tmp_path / "a.tif"
    _write_tile(str(p), 300000.0, 5500000.0)
    z1, m1, meta1 = loader.load_dem(str(p))
    z2, m2, meta2 = loader.load_mosaic([str(p)])
    assert z1.shape == z2.shape
    assert np.array_equal(m1, m2)


def test_mosaic_merges_adjacent_tiles(tmp_path):
    # two tiles side by side in the same CRS; merged width should roughly double
    a = tmp_path / "a.tif"; b = tmp_path / "b.tif"
    _write_tile(str(a), 300000.0, 5500000.0, nx=120)
    _write_tile(str(b), 300120.0, 5500000.0, nx=120)  # 120 m to the east
    z, mask, meta = loader.load_mosaic([str(a), str(b)])
    assert meta["width"] >= 230           # ~240 minus any overlap rounding
    assert meta["height"] == 120
    assert (~mask).sum() > 0


def test_mosaic_reprojects_mixed_crs(tmp_path):
    # real LidarBC neighbours mix EPSG:6654 and 2955; merge must not raise and
    # must produce a single-CRS grid (the first tile's CRS).
    a = tmp_path / "a.tif"; b = tmp_path / "b.tif"
    _write_tile(str(a), 300000.0, 5500000.0, crs="EPSG:2955")
    _write_tile(str(b), 300120.0, 5500000.0, crs="EPSG:6654")
    z, mask, meta = loader.load_mosaic([str(a), str(b)])
    assert meta["crs"] == rasterio.crs.CRS.from_epsg(2955)
    assert np.isfinite(z[~mask]).all()


def test_boundary_clip_masks_outside(tmp_path):
    p = tmp_path / "a.tif"
    _write_tile(str(p), 300000.0, 5500000.0, ny=120, nx=120)
    z, mask, meta = loader.load_dem(str(p))
    before = int((~mask).sum())

    # WGS84 polygon covering the WEST half of the tile's footprint. The tile
    # spans lon -119.7691..-119.7674; cut it at the midpoint -119.7682.
    poly = {"type": "Polygon", "coordinates": [[
        [-119.7691, 49.6205], [-119.7682, 49.6205],
        [-119.7682, 49.6180], [-119.7691, 49.6180], [-119.7691, 49.6205]]]}
    zc, maskc = loader.clip_to_boundary(z, mask, meta, poly)
    after = int((~maskc).sum())
    assert 0 < after < before                 # kept the west half only
    assert after == pytest.approx(before * 0.5, rel=0.15)
    assert np.isnan(zc[maskc]).all()


def test_boundary_clip_accepts_feature(tmp_path):
    p = tmp_path / "a.tif"
    _write_tile(str(p), 300000.0, 5500000.0)
    z, mask, meta = loader.load_dem(str(p))
    feature = {"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[
            [-119.7691, 49.6205], [-119.7674, 49.6205],
            [-119.7674, 49.6180], [-119.7691, 49.6180],
            [-119.7691, 49.6205]]]}}
    _, maskc = loader.clip_to_boundary(z, mask, meta, feature)
    assert (~maskc).sum() > 0          # a Feature wrapper is unwrapped, not dropped


def test_boundary_clip_bad_geojson_is_noop(tmp_path):
    p = tmp_path / "a.tif"
    _write_tile(str(p), 300000.0, 5500000.0)
    z, mask, meta = loader.load_dem(str(p))
    zc, maskc = loader.clip_to_boundary(z, mask, meta, {"type": "Nonsense"})
    assert np.array_equal(mask, maskc)  # unrecognized boundary changes nothing


def test_run_accepts_multiple_paths(tmp_path):
    a = tmp_path / "bc_a_dtm.tif"; b = tmp_path / "bc_b_dtm.tif"
    _write_tile(str(a), 300000.0, 5500000.0)
    _write_tile(str(b), 300120.0, 5500000.0)
    out = tmp_path / "job"
    stats = run.run([str(a), str(b)], str(out))
    assert (out / "stats.json").is_file()
    assert stats["coverage"]["valid_ha"] > 0
    import json
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tile_count"] == 2
    assert isinstance(meta["filename"], list)
