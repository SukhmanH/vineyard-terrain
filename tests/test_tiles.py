"""Tile index, delivery choice and AOI mosaics (backend/pipeline/tiles.py, net.py).

Bucket listings come from tests/fixtures/bucket/ (trimmed real responses). Tiles are small
synthetic GeoTIFFs in real NAD83(CSRS) UTM 11N coordinates near Oliver, so footprints, CRS
handling and the AOI grid run exactly as they do on LidarBC data.
"""
import hashlib
import os

import numpy as np
import pytest
import rasterio

from backend.pipeline import net, tiles
from tests.helpers import X0, Y0, entry_for, utm_box, write_tile

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "bucket")
ND = -32767.0


def _fixture(name):
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


def plane(X, Y, base=300.0, gx=0.10, gy=0.02):
    return base + gx * (X - X0) + gy * (Y0 - Y)


@pytest.fixture
def local_fetch(monkeypatch):
    paths = {}
    monkeypatch.setattr(tiles, "fetch", lambda e, window=None: paths[e["key"]])
    return paths


# ---------------------------------------------------------------- listing
def test_list_keys_pages_on_the_last_key(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(dict(params))
        return _fixture("dem_2024_page2.xml" if "marker" in params else "dem_2024_page1.xml")

    monkeypatch.setattr(net, "get", fake_get)
    keys = tiles.list_keys("082/082e/2024/dem/")
    assert [k["key"].split("/")[-1] for k in keys] == [
        "_$folder$",
        "bc_082e002_1_1_2_xli1m_utm11_20240721_20250719.tif",
        "bc_082e002_1_1_4_xli1m_utm11_20240721_20250719.tif",
        "bc_082e002_1_2_2_xli1m_utm11_20240725_20250717.tif",
        "bc_082e002_1_2_4_xli1m_utm11_20240725_20250717.tif",
    ]
    # v1 listings without a delimiter carry no NextMarker: the last key is the marker
    assert calls[1]["marker"].endswith("bc_082e002_1_1_4_xli1m_utm11_20240721_20250719.tif")
    assert keys[1]["size"] == 8786803 and not keys[1]["etag"].startswith('"')


def test_list_prefixes_reads_delivery_years(monkeypatch):
    monkeypatch.setattr(net, "get", lambda url, params=None, **kw: _fixture("years_082e.xml"))
    years = [p.split("/")[-2] for p in tiles.list_prefixes("082/082e/")]
    assert years == ["2017", "2018", "2019", "2020", "2022", "2024", "2025"]


def test_parse_key_takes_the_product_from_the_path():
    dem = tiles.parse_key("082/082e/2024/dem/bc_082e023_1_2_1_xli1m_utm11_20240721_20240726.tif")
    assert (dem["year"], dem["product"], dem["stem"]) == (2024, "dem", "bc_082e023_1_2_1")
    dsm = tiles.parse_key("082/082e/2024/dsm/bc_082e023_1_2_1_xli1m_utm11_20240721_20240726_dsm.tif")
    assert dsm["product"] == "dsm"
    copy = tiles.parse_key("082/082e/2018/dem/bc_082e003_xli1m_utm11_2018 (2).tif")
    assert (copy["year"], copy["stem"]) == (2018, "bc_082e003")    # sheet-only name
    for not_a_tile in ("082/082e/2024/dem/_$folder$", "082/082e/2018/orthophoto/x.tif",
                       "082/082e/2024/pointcloud/x.laz", "../../etc/passwd"):
        assert tiles.parse_key(not_a_tile) is None


def test_remote_path_only_accepts_listing_shaped_keys():
    assert tiles.remote_path("082/082e/2018/dem/bc_082e003_xli1m_utm11_2018 (2).tif") \
        .startswith("/vsicurl/https://nrs.objectstore.gov.bc.ca/gdwuts/082/082e/2018/dem/")
    with pytest.raises(tiles.TileError):
        tiles.remote_path("https://evil.example/x.tif")


# ------------------------------------------------------------------ index
def test_refresh_index_reads_dem_headers_and_is_incremental(tmp_path, monkeypatch):
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path / "data"))
    a = write_tile(str(tmp_path / "a.tif"), X0, Y0, 50, 40, plane)
    b = write_tile(str(tmp_path / "b.tif"), X0 + 50, Y0, 50, 40, plane)
    listing = {
        "082/082e/2024/dem/": [
            {"key": "082/082e/2024/dem/_$folder$", "size": 0, "etag": "e", "last_modified": ""},
            {"key": "082/082e/2024/dem/bc_082e023_1_2_1_xli1m_utm11_x.tif", "size": 1,
             "etag": "e1", "last_modified": "2026-01-01"},
            {"key": "082/082e/2024/dem/bc_082e023_1_2_2_xli1m_utm11_x.tif", "size": 1,
             "etag": "e2", "last_modified": "2026-01-01"}],
        "082/082e/2024/dsm/": [
            {"key": "082/082e/2024/dsm/bc_082e023_1_2_1_xli1m_utm11_x_dsm.tif", "size": 1,
             "etag": "e3", "last_modified": "2026-01-01"}],
    }
    local = {"bc_082e023_1_2_1_xli1m_utm11_x.tif": a, "bc_082e023_1_2_2_xli1m_utm11_x.tif": b}
    opened = []

    def fake_open(key):
        opened.append(key)
        return rasterio.open(local[key.split("/")[-1]])

    monkeypatch.setattr(tiles, "list_prefixes", lambda p: ["082/082e/2024/", "082/082e/x/"])
    monkeypatch.setattr(tiles, "list_keys", lambda p: listing.get(p, []))
    monkeypatch.setattr(tiles, "_open_remote", fake_open)

    doc = tiles.refresh_index("082e")
    dem = [e for e in doc["entries"] if e["product"] == "dem"]
    dsm = [e for e in doc["entries"] if e["product"] == "dsm"]
    assert len(dem) == 2 and len(dsm) == 1          # folder marker skipped
    assert all(len(e["footprint_3005"]) == 4 and e["crs"] == "EPSG:2955" for e in dem)
    assert "footprint_3005" not in dsm[0]           # DSM headers read only when fetched
    assert len(opened) == 2

    opened.clear()
    tiles.refresh_index("082e")                     # same ETags: nothing re-read
    assert opened == []
    assert os.path.exists(tiles.index_path("082e"))


def test_unreadable_header_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tiles, "list_prefixes", lambda p: ["082/082e/2024/"])
    monkeypatch.setattr(tiles, "list_keys", lambda p: [
        {"key": "082/082e/2024/dem/bc_082e023_1_2_1_x.tif", "size": 1, "etag": "e",
         "last_modified": ""}] if p.endswith("dem/") else [])

    def broken(key):
        raise rasterio.errors.RasterioIOError("HTTP 503")

    monkeypatch.setattr(tiles, "_open_remote", broken)
    doc = tiles.refresh_index("082e")
    assert len(doc["header_failures"]) == 1 and "503" in doc["header_failures"][0]


# -------------------------------------------------------------- selection
def _square_entry(year, name, x0, y0, size, product="dem", lm="2026-01-01", crs="EPSG:2955"):
    e = tiles.parse_key(f"082/082e/{year}/{product}/{name}")
    e.update(crs=crs, bounds=[x0, y0, x0 + size, y0 + size], width=int(size),
             height=int(size), res=1.0, nodata=ND, size=1, etag="e", last_modified=lm,
             footprint_3005=[list(p) for p in utm_box(x0, y0, x0 + size, y0 + size)
                             .exterior.coords[:4]])
    return e


def test_dedupe_keeps_one_tile_per_footprint_newest_upload():
    old = _square_entry(2018, "bc_082e003_xli1m_utm11_2018.tif", X0, Y0, 100, lm="2024-07-01")
    new = _square_entry(2018, "bc_082e003_xli1m_utm11_2018 (2).tif", X0, Y0, 100, lm="2024-07-09")
    other = _square_entry(2018, "bc_082e004_xli1m_utm11_2018.tif", X0 + 100, Y0, 100)
    kept = tiles.dedupe([old, new, other])
    assert {k["name"] for k in kept} == {new["name"], other["name"]}


def test_newest_delivery_that_covers_the_aoi_wins():
    aoi = utm_box(X0 + 10, Y0 + 10, X0 + 90, Y0 + 90)
    entries = [
        _square_entry(2017, "bc_082e023_1_2_1_xl1m_17603.tif", X0, Y0, 100),
        _square_entry(2024, "bc_082e023_1_2_1_xli1m_utm11_2024.tif", X0, Y0, 100),
        _square_entry(2025, "bc_082e023_1_2_1_xli1m_utm11_2025.tif", X0, Y0, 50),   # partial
        _square_entry(2024, "bc_082e023_1_2_1_xli1m_utm11_2024_dsm.tif", X0, Y0, 100,
                      product="dsm"),
    ]
    deliveries = tiles.deliveries_for(entries, aoi)
    assert [d["year"] for d in deliveries] == [2025, 2024, 2017]
    assert all(t["product"] == "dem" for d in deliveries for t in d["tiles"])
    choice = tiles.choose(deliveries)
    assert choice["primary"]["year"] == 2024
    assert [d["year"] for d in choice["alternates"]] == [2017]
    assert choice["fill"] == []


def test_mixing_only_when_no_delivery_covers_the_aoi():
    aoi = utm_box(X0 + 10, Y0 + 10, X0 + 190, Y0 + 90)
    entries = [
        _square_entry(2025, "bc_082e023_1_2_1_x.tif", X0, Y0, 100),
        _square_entry(2017, "bc_082e023_1_2_2_x.tif", X0 + 100, Y0, 100),
    ]
    choice = tiles.choose(tiles.deliveries_for(entries, aoi))
    assert choice["primary"]["year"] == 2025          # equal cover: newer leads
    assert [d["year"] for d in choice["fill"]] == [2017]


def test_dsm_twin_is_found_by_delivery_and_stem():
    dem = _square_entry(2024, "bc_082e023_1_2_1_xli1m_utm11_20240721_20240726.tif", X0, Y0, 100)
    twin = _square_entry(2024, "bc_082e023_1_2_1_xli1m_utm11_20240721_20240726_dsm.tif",
                         X0, Y0, 100, product="dsm")
    other_year = _square_entry(2025, "bc_082e023_1_2_1_xli1m_utm11_2025_dsm.tif", X0, Y0, 100,
                               product="dsm")
    assert tiles.dsm_twin([other_year, twin, dem], dem) is twin


# ---------------------------------------------------------------- mosaics
def test_mosaic_is_cropped_to_the_aoi_not_the_union_of_tiles(tmp_path, local_fetch):
    """Job a2a89378 merged tiles ~40 km apart into one 417 MB union bbox."""
    near = write_tile(str(tmp_path / "near.tif"), X0, Y0, 400, 300, plane)
    far = write_tile(str(tmp_path / "far.tif"), X0 + 40000, Y0, 400, 300, plane)
    e_near = entry_for(near, 2024, "bc_082e023_1_2_1_x.tif")
    e_far = entry_for(far, 2024, "bc_082e025_1_2_1_x.tif")
    local_fetch.update({e_near["key"]: near, e_far["key"]: far})
    aoi = utm_box(X0 + 100, Y0 - 250, X0 + 300, Y0 - 50)
    parcel = utm_box(X0 + 180, Y0 - 170, X0 + 220, Y0 - 130)
    z, mask, meta, info = tiles.load_aoi(aoi, parcel, [e_near, e_far])
    assert z.shape[1] < 260 and z.shape[0] < 260        # ~200 m AOI, never the union
    assert info["tile_keys"] == [e_near["key"]]
    assert info["parcel_valid_pct"] == 100.0
    assert np.isfinite(z[~mask]).all()


def test_one_delivery_per_footprint(tmp_path, local_fetch):
    """Job d3c70984 stacked DSM and DEM tiles of two years on the same footprints and got a
    mean slope of 84.7 %. Same footprint in two DEM deliveries: only the newest is used."""
    t17 = write_tile(str(tmp_path / "t17.tif"), X0, Y0, 300, 300, plane)
    t24 = write_tile(str(tmp_path / "t24.tif"), X0, Y0, 300, 300,
                     lambda X, Y: plane(X, Y) + 0.4)
    dsm = write_tile(str(tmp_path / "dsm.tif"), X0, Y0, 300, 300,
                     lambda X, Y: plane(X, Y) + 12.0)
    e17 = entry_for(t17, 2017, "bc_082e023_1_2_1_xl1m_17603.tif")
    e24 = entry_for(t24, 2024, "bc_082e023_1_2_1_xli1m_utm11_x.tif")
    edsm = entry_for(dsm, 2024, "bc_082e023_1_2_1_xli1m_utm11_x_dsm.tif", product="dsm")
    local_fetch.update({e17["key"]: t17, e24["key"]: t24, edsm["key"]: dsm})
    aoi = utm_box(X0 + 20, Y0 - 280, X0 + 280, Y0 - 20)
    parcel = utm_box(X0 + 120, Y0 - 180, X0 + 180, Y0 - 120)
    z, mask, meta, info = tiles.load_aoi(aoi, parcel, [e17, e24, edsm])
    assert info["years"] == [2024] and info["tile_keys"] == [e24["key"]]
    assert info["product"] == "dem" and info["mix"] is None
    gy, gx = np.gradient(z[5:-5, 5:-5])
    assert np.nanmax(np.hypot(gx, gy)) < 0.2         # a plane, not a stack


def test_older_delivery_is_used_when_the_newest_misses_the_parcel(tmp_path, local_fetch):
    """Some LidarBC DEMs are mostly NoData by design (bare earth for the valley only)."""
    new = write_tile(str(tmp_path / "new.tif"), X0, Y0, 300, 300, plane,
                     nodata_where=lambda X, Y: X < X0 + 200)
    old = write_tile(str(tmp_path / "old.tif"), X0, Y0, 300, 300, plane)
    e_new = entry_for(new, 2025, "bc_082e023_1_2_1_x.tif")
    e_old = entry_for(old, 2017, "bc_082e023_1_2_1_xl1m.tif")
    local_fetch.update({e_new["key"]: new, e_old["key"]: old})
    aoi = utm_box(X0 + 20, Y0 - 280, X0 + 280, Y0 - 20)
    parcel = utm_box(X0 + 100, Y0 - 180, X0 + 160, Y0 - 120)
    z, mask, meta, info = tiles.load_aoi(aoi, parcel, [e_new, e_old])
    assert info["years"] == [2017]
    assert [c["year"] for c in info["deliveries_considered"]] == [2025, 2017]
    assert info["deliveries_considered"][0]["parcel_valid_pct"] == 0.0
    assert info["parcel_valid_pct"] == 100.0


def _two_deliveries(tmp_path, local_fetch, step, overlap):
    """A 2025 tile on the west and a 2017 tile on the east, abutting or overlapping, on one
    tilted plane plus a vertical step in the 2017 delivery."""
    rng = np.random.default_rng(1)
    noise = lambda X: rng.normal(0.0, 0.02, X.shape)        # LiDAR-scale noise
    west_nx = 300 + (100 if overlap else 0)
    west = write_tile(str(tmp_path / "w.tif"), X0, Y0, west_nx, 300,
                      lambda X, Y: plane(X, Y) + noise(X))
    east = write_tile(str(tmp_path / "e.tif"), X0 + 300, Y0, 300, 300,
                      lambda X, Y: plane(X, Y) + step + noise(X))
    ew = entry_for(west, 2025, "bc_082e023_1_2_1_x.tif")
    ee = entry_for(east, 2017, "bc_082e023_1_2_2_x.tif")
    local_fetch.update({ew["key"]: west, ee["key"]: east})
    aoi = utm_box(X0 + 20, Y0 - 280, X0 + 580, Y0 - 20)
    parcel = utm_box(X0 + 100, Y0 - 180, X0 + 160, Y0 - 120)
    return tiles.load_aoi(aoi, parcel, [ew, ee])


@pytest.mark.parametrize("overlap", [False, True])
def test_seam_step_over_tolerance_refuses_the_mix(tmp_path, local_fetch, overlap):
    z, mask, meta, info = _two_deliveries(tmp_path, local_fetch, 0.30, overlap)
    assert info["mix"]["refused"] is True
    assert info["mix"]["offset_m"] == pytest.approx(0.30, abs=0.03)
    assert "over 0.15 m" in info["mix"]["reason"]
    assert info["years"] == [2025]
    assert mask[:, -50:].all()                      # east half left out, not stitched in


@pytest.mark.parametrize("overlap", [False, True])
def test_seam_step_within_tolerance_mixes(tmp_path, local_fetch, overlap):
    z, mask, meta, info = _two_deliveries(tmp_path, local_fetch, 0.05, overlap)
    assert info["mix"]["refused"] is False
    assert info["mix"]["method"] == ("overlap" if overlap else "across")
    assert info["mix"]["offset_m"] == pytest.approx(0.05, abs=0.03)
    assert info["years"] == [2025, 2017]
    assert (~mask).mean() > 0.99


def test_a_slope_alone_is_not_read_as_a_seam_step():
    """Cells either side of a seam on a 20 % slope differ by 0.2 m with no step at all."""
    yy, xx = np.mgrid[0:200, 0:200].astype("float64")
    z = 300.0 + 0.2 * xx
    a = np.where(xx < 100, z, np.nan)
    b = np.where(xx >= 100, z, np.nan)
    s = tiles.seam_step(a, b, 1.0)
    assert s["method"] == "across" and abs(s["step_m"]) < 0.01


def test_seam_with_no_overlap_or_contact_is_unmeasurable():
    a = np.full((50, 50), np.nan); a[:, :10] = 300.0
    b = np.full((50, 50), np.nan); b[:, 40:] = 300.0
    assert tiles.seam_step(a, b, 1.0) is None


def test_other_crs_tiles_are_warped_onto_the_aoi_grid(tmp_path, local_fetch):
    """Neighbouring LidarBC deliveries mix EPSG:26911, 2955 and 6654."""
    t = write_tile(str(tmp_path / "nad83.tif"), X0, Y0, 300, 300, plane, crs="EPSG:26911")
    e = entry_for(t, 2017, "bc_082e023_1_2_1_xl1m_17603.tif")
    grid = tiles.plan_grid(utm_box(X0 + 20, Y0 - 280, X0 + 280, Y0 - 20), "EPSG:2955")
    z = tiles.assemble([t], grid)
    assert np.isfinite(z).mean() > 0.95
    assert np.nanmax(np.abs(np.gradient(z[5:-5, 5:-5])[1])) < 0.2


def test_an_aoi_over_the_cell_cap_drops_to_2_m(monkeypatch):
    monkeypatch.setattr(tiles, "MAX_AOI_CELLS", 100_000)
    aoi = utm_box(X0, Y0 - 500, X0 + 500, Y0)        # 250,000 cells at 1 m
    grid = tiles.plan_grid(aoi, "EPSG:2955")
    assert grid["res"] == 2.0 and grid["width"] * grid["height"] <= 100_000
    monkeypatch.setattr(tiles, "MAX_AOI_CELLS", 1_000)
    with pytest.raises(tiles.TileError, match="cell cap"):
        tiles.plan_grid(aoi, "EPSG:2955")


def test_no_dem_coverage_is_a_clear_error():
    aoi = utm_box(X0, Y0 - 100, X0 + 100, Y0)
    with pytest.raises(tiles.TileError, match="no LidarBC bare-earth"):
        tiles.load_aoi(aoi, aoi, [])


# ------------------------------------------------------------------ cache
def test_eviction_drops_least_recent_and_keeps_in_use(tmp_path, monkeypatch):
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path))
    paths = []
    for i in range(3):
        p = tiles.cache_path(f"082/082e/2024/dem/bc_082e023_1_2_{i + 1}_x.tif")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x" * 1024 * 1024)
        os.utime(p, (1000 + i, 1000 + i))
        paths.append(p)
    # 3 MB cached, 2 MB cap: the oldest file not in use goes, and only that one
    removed = tiles.evict(keep=[paths[0]], cap_mb=2)
    assert os.path.exists(paths[0]) and os.path.exists(paths[2])
    assert removed == [os.path.abspath(paths[1])]
    assert tiles.evict(keep=[paths[0]], cap_mb=0) == [os.path.abspath(paths[2])]
    assert os.path.exists(paths[0])


def test_mapsheet_sized_files_are_read_by_window_not_downloaded(tmp_path, monkeypatch):
    """2019 DEMs are whole 1:20,000 mapsheets (~709 MB); only the AOI window is read."""
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path / "data"))
    sheet = write_tile(str(tmp_path / "sheet.tif"), X0, Y0, 1200, 900, plane)
    e = entry_for(sheet, 2019, "bc_082e023_xli1m_utm11_2019.tif")
    e["size"] = tiles.DOWNLOAD_MAX_BYTES + 1
    monkeypatch.setattr(tiles, "_open_remote", lambda key: rasterio.open(sheet))
    monkeypatch.setattr(net, "download", lambda *a, **k: pytest.fail("downloaded whole"))
    window = utm_box(X0 + 400, Y0 - 500, X0 + 600, Y0 - 300).bounds
    path = tiles.fetch(e, window)
    with rasterio.open(path) as ds:
        assert 200 <= ds.width < 400 and 200 <= ds.height < 400   # window plus margin
        assert ds.crs.to_epsg() == 2955
    assert tiles.fetch(e, window) == path                          # cached by window
    with pytest.raises(tiles.TileError, match="no AOI window"):
        tiles.fetch(e)


def test_fetch_uses_the_cache_when_size_and_etag_match(tmp_path, monkeypatch):
    monkeypatch.setenv("VTA_DATA_DIR", str(tmp_path))
    e = {"key": "082/082e/2024/dem/bc_082e023_1_2_1_x.tif", "size": 3, "etag": "abc"}
    calls = []

    def fake_download(url, dest, max_bytes, expect_size=None, expect_md5=None):
        calls.append(url)
        with open(dest, "wb") as f:
            f.write(b"xyz")
        return 3

    monkeypatch.setattr(net, "download", fake_download)
    p1 = tiles.fetch(e)
    p2 = tiles.fetch(e)
    assert p1 == p2 and len(calls) == 1
    assert calls[0] == net.BUCKET + "/082/082e/2024/dem/bc_082e023_1_2_1_x.tif"
    e["etag"] = "changed upstream"
    tiles.fetch(e)
    assert len(calls) == 2


# -------------------------------------------------------------------- net
def test_only_the_three_allowed_hosts_are_reachable():
    for ok in (net.BUCKET + "/x.tif", net.WFS, net.GEOCODER):
        assert net.check_url(ok) == ok
    for bad in ("http://openmaps.gov.bc.ca/geo/pub/wfs", "https://evil.example/x",
                "https://openmaps.gov.bc.ca.evil.example/x", "ftp://ftp.nrcan.gc.ca/x",
                "file:///etc/passwd"):
        with pytest.raises(net.NetError):
            net.check_url(bad)


def test_redirects_are_refused():
    with pytest.raises(net.NetError, match="redirect"):
        net._NoRedirect().redirect_request(None, None, 302, "Found", {},
                                           "https://evil.example/")


class _Resp:
    def __init__(self, body):
        self.body, self.pos = body, 0

    def read(self, n=-1):
        n = len(self.body) - self.pos if n < 0 else n
        chunk = self.body[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_lands_only_when_size_and_checksum_match(tmp_path, monkeypatch):
    monkeypatch.setattr(net._opener, "open", lambda req, timeout=None: _Resp(b"tile bytes"))
    dest = str(tmp_path / "t.tif")
    good = hashlib.md5(b"tile bytes").hexdigest()
    with pytest.raises(net.NetError, match="checksum"):
        net.download(net.BUCKET + "/k.tif", dest, 100, expect_size=10, expect_md5="0" * 32)
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")
    with pytest.raises(net.NetError, match="short"):
        net.download(net.BUCKET + "/k.tif", dest, 100, expect_size=11)
    assert net.download(net.BUCKET + "/k.tif", dest, 100, expect_size=10,
                        expect_md5=good) == 10
    assert open(dest, "rb").read() == b"tile bytes"


def test_get_caps_the_response_size(monkeypatch):
    monkeypatch.setattr(net._opener, "open", lambda req, timeout=None: _Resp(b"x" * 50))
    with pytest.raises(net.NetError, match="over 10 bytes"):
        net.get(net.WFS, {"a": 1}, max_bytes=10)
