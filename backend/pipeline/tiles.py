"""LidarBC tile index, delivery choice, fetch and AOI mosaics for the parcel path.

A listing is a parcel; LidarBC is BCGS 1:2,500 tiles of about 280 ha, delivered more than
once (2017, 2024 and 2025 in the south Okanagan) in several naming styles. The rules this
module holds, each learned from a job that went wrong:

  - Bare-earth `dem` only, never mixed with `dsm`. The product comes from the bucket path
    (<year>/dem/ vs <year>/dsm/): DEM file names carry no _dtm suffix, so the file name
    cannot tell a DEM from a DSM.
  - Each footprint once, from one delivery. Job d3c70984 stacked DSM and DEM tiles of two
    delivery years on the same footprints and reported a mean slope of 84.7 %.
  - The newest delivery that covers the whole AOI wins. Deliveries mix only when none does,
    and only across a seam whose elevation step is within SEAM_MAX_STEP_M: closed_hollow
    reacts at 0.1 m, so a seam step reads as a dam or a hollow. 2024/25 tiles are on
    CGVD2013 heights (EPSG:6654) and 2017 tiles carry no vertical datum, so expect steps.
  - The mosaic is the AOI grid, never the union bbox of the tiles. Job a2a89378 merged
    tiles from three mapsheets ~40 km apart into a 417 MB DEM and never wrote stats.

Footprints come from each GeoTIFF's header, read over /vsicurl/, not from the file name:
2018 names are sheet-only with " (2)" copies, and 2019 and 2022 files are whole 1:20,000
mapsheets. See docs/data-sources.md for what the bucket actually holds.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from . import net

ND = -32767.0
PRODUCTS = ("dem", "dsm")
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
KEY_RE = re.compile(r"^(?P<block>\d{3})/(?P<sheet>\d{3}[a-p])/(?P<year>\d{4})/"
                    r"(?P<product>dem|dsm)/(?P<name>[A-Za-z0-9_ ().-]+\.tif)$", re.I)
STEM_RE = re.compile(r"^(bc_\d{3}[a-p]\d{3}(?:_\d_\d_\d)?)_", re.I)
HEADER_KEYS = ("crs", "bounds", "width", "height", "res", "nodata", "footprint_3005")

INDEX_MAX_AGE_DAYS = 30     # an empty candidate set refreshes an index older than this
HEADER_WORKERS = 8
FULL_COVER_FRAC = 0.995     # footprints covering this much of the AOI count as covering it
DEDUPE_M = 0.5              # same CRS and bounds this close = the same tile delivered twice
MIN_PARCEL_VALID_PCT = 80.0  # below this, try an older full-cover delivery too
DOWNLOAD_MAX_BYTES = 64 * 1024 * 1024   # bigger files (2019 mapsheets) are read by window
WINDOW_MARGIN_M = 50.0
DEFAULT_CACHE_MB = 4000
MAX_AOI_CELLS = 11_000_000  # the four-tile mosaic the 2 GB VM already handles
COARSE_RES_M = 2.0          # resolution an over-cap AOI drops to
SEAM_MAX_STEP_M = 0.15
SEAM_STRIP_M = 30.0         # overlap measured within this distance of the seam
SEAM_MIN_CELLS = 2000       # overlap cells needed to trust the overlap median
SEAM_MIN_PAIRS = 200        # cross-seam pairs needed when deliveries only abut
SEAM_GRAD_CELLS = 3         # cells back from the seam used for each side's gradient

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TIMEOUT": "60",
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TileError(RuntimeError):
    """No usable bare-earth coverage, or a tile that could not be fetched."""


# ------------------------------------------------------------------- locations
def data_dir():
    """VTA_DATA_DIR (a mounted volume in a container) or the repo root, read per call so
    tests and the API agree on where tiles live."""
    return os.environ.get("VTA_DATA_DIR") or ROOT


def tiles_dir(*parts):
    return os.path.join(data_dir(), "tiles", *parts)


def configured_sheets():
    return [s.strip().lower() for s in os.environ.get("VTA_SHEETS", "082e").split(",")
            if s.strip()]


def index_path(sheet):
    return tiles_dir("index", f"{sheet.lower()}.json")


def cache_path(key):
    return tiles_dir("cache", *key.split("/"))


# --------------------------------------------------------------------- listing
def parse_key(key):
    """Split a bucket key into its parts, or None for anything that is not a DEM/DSM tif
    (folder markers like `_$folder$`, orthophotos, point clouds)."""
    m = KEY_RE.match(key)
    if not m:
        return None
    stem = STEM_RE.match(m["name"])
    return {
        "key": key,
        "block": m["block"],
        "sheet": m["sheet"].lower(),
        "year": int(m["year"]),
        "product": m["product"].lower(),
        "name": m["name"],
        "stem": stem.group(1).lower() if stem else None,
    }


def _list_page(params):
    return ET.fromstring(net.get(net.BUCKET, params))


def list_prefixes(prefix):
    """Immediate sub-prefixes ("folders") under prefix."""
    root = _list_page({"prefix": prefix, "delimiter": "/"})
    return [p.findtext("s:Prefix", namespaces=S3_NS)
            for p in root.findall("s:CommonPrefixes", S3_NS)]


def list_keys(prefix):
    """Every object under prefix. ListObjects v1 pages at 1,000 keys and only returns
    NextMarker when a delimiter is set, so the last key of a page is the next marker."""
    out, marker = [], ""
    while True:
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        root = _list_page(params)
        page = [{
            "key": c.findtext("s:Key", namespaces=S3_NS),
            "size": int(c.findtext("s:Size", default="0", namespaces=S3_NS)),
            "etag": (c.findtext("s:ETag", default="", namespaces=S3_NS) or "").strip('"'),
            "last_modified": c.findtext("s:LastModified", default="", namespaces=S3_NS),
        } for c in root.findall("s:Contents", S3_NS)]
        out.extend(page)
        truncated = root.findtext("s:IsTruncated", default="false", namespaces=S3_NS)
        if truncated != "true" or not page:
            return out
        marker = root.findtext("s:NextMarker", namespaces=S3_NS) or page[-1]["key"]


# --------------------------------------------------------------------- headers
def remote_path(key):
    """GDAL path for a bucket key. Only keys shaped like our own listing get through."""
    if not KEY_RE.match(key):
        raise TileError(f"not a LidarBC DEM/DSM key: {key!r}")
    return "/vsicurl/" + net.BUCKET + "/" + urllib.parse.quote(key)


def _open_remote(key):
    return rasterio.open(remote_path(key))


def header_of(ds):
    """What the index keeps from a GeoTIFF header: CRS, native bounds and size, and the
    footprint as four corners in EPSG:3005 (BC Albers, metric) for AOI intersection."""
    epsg = ds.crs.to_epsg() if ds.crs else None
    b = ds.bounds
    xs = [b.left, b.right, b.right, b.left]
    ys = [b.top, b.top, b.bottom, b.bottom]
    fx, fy = transform(ds.crs, "EPSG:3005", xs, ys)
    return {
        "crs": f"EPSG:{epsg}" if epsg else ds.crs.to_wkt(),
        "bounds": [b.left, b.bottom, b.right, b.top],
        "width": ds.width,
        "height": ds.height,
        "res": float(ds.res[0]),
        "nodata": ds.nodata,
        "footprint_3005": [[round(x, 1), round(y, 1)] for x, y in zip(fx, fy)],
    }


def read_header(key):
    with rasterio.Env(**GDAL_ENV), _open_remote(key) as ds:
        return header_of(ds)


# ----------------------------------------------------------------------- index
def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def refresh_index(sheet, workers=HEADER_WORKERS, log=None):
    """List a sheet's DEM and DSM tiles and read every new or changed DEM header.

    Incremental by ETag, so a refresh after a new delivery only reads the new tiles. DSM
    keys are listed without headers: they are only ever twins of a DEM (canopy height),
    matched by delivery and stem and checked when fetched.
    """
    sheet = sheet.lower()
    block = sheet[:3]
    path = index_path(sheet)
    old = {}
    if os.path.exists(path):
        old = {e["key"]: e for e in _read_json(path).get("entries", [])}

    objects = []
    for year_prefix in list_prefixes(f"{block}/{sheet}/"):
        year = year_prefix.rstrip("/").split("/")[-1]
        if not year.isdigit():
            continue
        for product in PRODUCTS:
            objects.extend(list_keys(f"{year_prefix}{product}/"))

    entries, todo = [], []
    for obj in objects:
        e = parse_key(obj["key"])
        if e is None:
            continue
        e.update(size=obj["size"], etag=obj["etag"], last_modified=obj["last_modified"])
        prev = old.get(e["key"])
        if e["product"] == "dem":
            if prev and prev.get("etag") == e["etag"] and prev.get("footprint_3005"):
                e.update({k: prev[k] for k in HEADER_KEYS})
            else:
                todo.append(e)
        entries.append(e)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_header, e["key"]): e for e in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            e = futures[fut]
            try:
                e.update(fut.result())
            except Exception as exc:   # one unreadable header must not sink the index
                failures.append(f"{e['key']}: {exc}")
            if log and i % 250 == 0:
                log(f"  headers {i}/{len(todo)}")

    doc = {
        "sheet": sheet,
        "bucket": net.BUCKET,
        "built_at": _utcnow(),
        "entries": sorted(entries, key=lambda e: e["key"]),
        "header_failures": failures,
    }
    _write_json(path, doc)
    if log:
        n_dem = sum(1 for e in entries if e["product"] == "dem")
        log(f"{sheet}: {n_dem} DEM + {len(entries) - n_dem} DSM tiles, "
            f"{len(todo)} headers read, {len(failures)} failed -> {path}")
    return doc


def index_age_days(sheet):
    path = index_path(sheet)
    if not os.path.exists(path):
        return None
    built = _read_json(path).get("built_at")
    when = datetime.strptime(built, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400.0


def load_index(sheets=None, build_missing=True):
    """All index entries for the configured sheets, building an index that is missing."""
    entries = []
    for sheet in sheets or configured_sheets():
        path = index_path(sheet)
        if not os.path.exists(path):
            if not build_missing:
                continue
            refresh_index(sheet)
        entries.extend(_read_json(path)["entries"])
    return entries


# ------------------------------------------------------------------- selection
def _footprint(entry):
    return Polygon(entry["footprint_3005"])


def _same_tile(a, b):
    return a["crs"] == b["crs"] and max(
        abs(p - q) for p, q in zip(a["bounds"], b["bounds"])) <= DEDUPE_M


def dedupe(tiles):
    """One tile per footprint inside a delivery: the newest upload of a footprint wins
    (the 2018 " (2)" copies are the same tile uploaded twice)."""
    kept = []
    for t in sorted(tiles, key=lambda t: t.get("last_modified") or "", reverse=True):
        if not any(_same_tile(k, t) for k in kept):
            kept.append(t)
    return kept


def deliveries_for(entries, aoi):
    """DEM deliveries whose tiles intersect aoi (shapely, EPSG:3005), newest first, each
    with its deduplicated tiles and the fraction of the AOI their footprints cover."""
    by_year = {}
    for e in entries:
        if e.get("product") != "dem" or not e.get("footprint_3005"):
            continue
        if _footprint(e).intersects(aoi):
            by_year.setdefault(e["year"], []).append(e)
    out = []
    for year in sorted(by_year, reverse=True):
        tiles = dedupe(by_year[year])
        cover = unary_union([_footprint(t) for t in tiles]).intersection(aoi).area / aoi.area
        out.append({"year": year, "tiles": tiles, "cover": round(float(cover), 4)})
    return out


def choose(deliveries):
    """Primary delivery, older full-cover alternates, and fill deliveries for a mix.

    A delivery whose footprints cover the AOI is always preferred to mixing. Only when none
    does, the best-covering delivery (newest on ties) leads and the others fill its gaps.
    """
    if not deliveries:
        return None
    full = [d for d in deliveries if d["cover"] >= FULL_COVER_FRAC]
    if full:
        return {"primary": full[0], "alternates": full[1:], "fill": []}
    ranked = sorted(deliveries, key=lambda d: (d["cover"], d["year"]), reverse=True)
    return {"primary": ranked[0], "alternates": [], "fill": ranked[1:]}


def dsm_twin(entries, dem_entry):
    """The DSM of the same delivery and footprint as a DEM tile, if the bucket has one."""
    if not dem_entry.get("stem"):
        return None
    for e in entries:
        if (e.get("product") == "dsm" and e["year"] == dem_entry["year"]
                and e.get("stem") == dem_entry["stem"]
                and e["name"].lower().startswith(dem_entry["name"][:-4].lower())):
            return e
    return None


# ----------------------------------------------------------------------- fetch
def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def fetch(entry, window_3005=None):
    """Local path of an index entry's tile, downloading it into the cache if needed.

    Tiles up to DOWNLOAD_MAX_BYTES are cached whole and checked against the listing's size
    and MD5 ETag. Bigger files (the ~709 MB 2019 mapsheets) are read only inside
    window_3005 (bounds in EPSG:3005) over /vsicurl/ and cached as a cropped GeoTIFF.
    """
    if entry["size"] > DOWNLOAD_MAX_BYTES:
        return _fetch_window(entry, window_3005)
    path = cache_path(entry["key"])
    etag_file = path + ".etag"
    if (os.path.exists(path) and os.path.getsize(path) == entry["size"]
            and _read_text(etag_file) == entry["etag"]):
        os.utime(path)    # recently used: last to be evicted
        return path
    md5 = entry["etag"] if re.fullmatch(r"[0-9a-f]{32}", entry["etag"] or "") else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        net.download(net.BUCKET + "/" + urllib.parse.quote(entry["key"]), path,
                     max_bytes=DOWNLOAD_MAX_BYTES, expect_size=entry["size"],
                     expect_md5=md5)
    except net.NetError as exc:
        raise TileError(f"could not fetch {entry['key']}: {exc}") from exc
    with open(etag_file, "w", encoding="utf-8") as f:
        f.write(entry["etag"])
    return path


def _fetch_window(entry, window_3005):
    if window_3005 is None:
        raise TileError(f"{entry['key']} is too big to cache whole and has no AOI window")
    native = transform_bounds("EPSG:3005", entry["crs"], *window_3005, densify_pts=21)
    m = WINDOW_MARGIN_M
    native = (native[0] - m, native[1] - m, native[2] + m, native[3] + m)
    tag = f"{entry['key']}|{entry['etag']}|" + ",".join(f"{v:.0f}" for v in native)
    path = tiles_dir("cache", "windows", hashlib.sha1(tag.encode()).hexdigest()[:20] + ".tif")
    if os.path.exists(path):
        os.utime(path)
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.Env(**GDAL_ENV), _open_remote(entry["key"]) as ds:
        win = from_bounds(*native, transform=ds.transform)
        win = win.round_offsets().round_lengths().intersection(
            Window(0, 0, ds.width, ds.height))
        data = ds.read(1, window=win)
        profile = {
            "driver": "GTiff", "width": int(win.width), "height": int(win.height),
            "count": 1, "dtype": data.dtype, "crs": ds.crs,
            "transform": ds.window_transform(win), "nodata": ds.nodata,
            "compress": "deflate", "tiled": True,
        }
    tmp = path + ".part"
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(data, 1)
    os.replace(tmp, path)
    return path


def evict(keep=(), cap_mb=None):
    """Drop the least recently used cached tiles until the cache is under its cap
    (VTA_TILE_CACHE_MB). Files in keep (the current job's) are never dropped."""
    cap = int(cap_mb if cap_mb is not None
              else os.environ.get("VTA_TILE_CACHE_MB", DEFAULT_CACHE_MB)) * 1024 * 1024
    keep = {os.path.abspath(p) for p in keep}
    files = []
    for dirpath, _, names in os.walk(tiles_dir("cache")):
        for name in names:
            if name.endswith(".tif"):
                p = os.path.abspath(os.path.join(dirpath, name))
                st = os.stat(p)
                files.append((st.st_mtime, st.st_size, p))
    total = sum(size for _, size, _ in files)
    removed = []
    for _, size, p in sorted(files):
        if total <= cap:
            break
        if p in keep:
            continue
        for q in (p, p + ".etag"):
            try:
                os.remove(q)
            except OSError:
                pass
        total -= size
        removed.append(p)
    return removed


# --------------------------------------------------------------------- mosaics
def aoi_grid(aoi, crs, res):
    """The analysis grid: the AOI's bounds in crs, snapped outward to whole cells.

    The AOI polygon itself is transformed, not its bbox: BC Albers and UTM 11N are rotated
    about 7 degrees against each other here, so a bbox of a bbox grows the grid twice.
    """
    crs = CRS.from_user_input(crs)
    left, bottom, right, top = shape(transform_geom("EPSG:3005", crs, mapping(aoi))).bounds
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
    return {
        "crs": crs, "res": float(res), "width": width, "height": height,
        "transform": from_origin(left, top, res, res),
        "bounds": BoundingBox(left, bottom, right, top),
    }


def plan_grid(aoi, crs, native_res=1.0):
    """AOI grid at native resolution, or COARSE_RES_M when that would pass MAX_AOI_CELLS."""
    grid = aoi_grid(aoi, crs, native_res)
    if grid["width"] * grid["height"] <= MAX_AOI_CELLS:
        return grid
    grid = aoi_grid(aoi, crs, COARSE_RES_M)
    if grid["width"] * grid["height"] > MAX_AOI_CELLS:
        raise TileError(f"AOI of {grid['width']} x {grid['height']} cells at "
                        f"{COARSE_RES_M:g} m is over the {MAX_AOI_CELLS:,} cell cap")
    return grid


def _aligned(ds, grid):
    """Same CRS and cell size, with an origin on the grid: merge can copy cells as is."""
    if ds.crs != grid["crs"] or abs(ds.res[0] - grid["res"]) > 1e-6:
        return False
    dx = (ds.transform.c - grid["transform"].c) / grid["res"]
    dy = (ds.transform.f - grid["transform"].f) / grid["res"]
    return abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6


def assemble(paths, grid):
    """One delivery's tiles on the AOI grid (NaN where there is no bare earth).

    This is rasterio.merge.merge(..., bounds=AOI): the output is the AOI and nothing else.
    Tiles in another CRS or off the grid go through a WarpedVRT built on the AOI grid, so
    each cell is resampled once (bilinear, or average when the AOI is coarsened).
    """
    coarse = grid["res"] > 1.0 + 1e-6
    opened, sources = [], []
    try:
        for p in paths:
            ds = rasterio.open(p)
            opened.append(ds)
            if _aligned(ds, grid):
                sources.append(ds)
                continue
            vrt = WarpedVRT(
                ds, crs=grid["crs"], transform=grid["transform"],
                width=grid["width"], height=grid["height"],
                resampling=Resampling.average if coarse else Resampling.bilinear,
                src_nodata=ds.nodata if ds.nodata is not None else ND, nodata=ND)
            opened.append(vrt)
            sources.append(vrt)
        arr, _ = merge(sources, bounds=tuple(grid["bounds"]), res=grid["res"], nodata=ND,
                       resampling=Resampling.average if coarse else Resampling.nearest)
    finally:
        for d in reversed(opened):
            d.close()
    z = arr[0].astype("float32")
    z[(z == ND) | ~np.isfinite(z)] = np.nan
    return z[:grid["height"], :grid["width"]]


def parcel_mask(parcel, grid):
    """True inside the parcel (shapely, EPSG:3005) on the grid."""
    geom = transform_geom("EPSG:3005", grid["crs"], mapping(parcel))
    return geometry_mask([geom], out_shape=(grid["height"], grid["width"]),
                         transform=grid["transform"], invert=True)


def valid_pct(z, inside):
    n = int(inside.sum())
    return 100.0 * float(np.isfinite(z[inside]).sum()) / n if n else 0.0


def _shift(a, dy, dx, pad):
    """View of a padded array offset by (dy, dx) cells."""
    h, w = a.shape[0] - 2 * pad, a.shape[1] - 2 * pad
    return a[pad + dy:pad + dy + h, pad + dx:pad + dx + w]


def seam_step(z_a, z_b, res):
    """Elevation step (delivery b minus delivery a) where b takes over from a.

    Where the two overlap near the seam, the step is the median difference over the
    overlap strip. Where they only abut, as LidarBC tiles do at a BCGS cell edge, it is the
    median over adjacent cell pairs across the seam, each side extrapolated to the seam with
    its own local gradient so a slope is not read as a step. Returns None when neither has
    enough cells to trust.
    """
    a, b = np.isfinite(z_a), np.isfinite(z_b)
    both = a & b
    if both.any():
        inside = ndimage.distance_transform_edt(a) * res
        strip = both & (inside <= SEAM_STRIP_M)
        n = int(strip.sum())
        if n >= SEAM_MIN_CELLS:
            return {"method": "overlap", "n": n,
                    "step_m": round(float(np.median(z_b[strip] - z_a[strip])), 3)}
    k = SEAM_GRAD_CELLS
    pad = k + 2
    za = np.pad(z_a.astype("float64"), pad, constant_values=np.nan)
    zb = np.pad(np.where(a, np.nan, z_b).astype("float64"), pad, constant_values=np.nan)
    steps = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        p = _shift(za, 0, 0, pad)
        p_back = _shift(za, -k * dy, -k * dx, pad)
        q = _shift(zb, dy, dx, pad)
        q_fwd = _shift(zb, (k + 1) * dy, (k + 1) * dx, pad)
        grad = ((p - p_back) + (q_fwd - q)) / (2.0 * k)   # rise per cell along the normal
        s = (q - p) - grad
        steps.append(s[np.isfinite(s)])
    steps = np.concatenate(steps)
    if steps.size >= SEAM_MIN_PAIRS:
        return {"method": "across", "n": int(steps.size),
                "step_m": round(float(np.median(steps)), 3)}
    return None


def _delivery_paths(delivery, aoi):
    return [fetch(t, aoi.bounds) for t in delivery["tiles"]]


def load_aoi(aoi, parcel, entries=None):
    """Bare-earth mosaic of aoi (shapely, EPSG:3005) for analyzing parcel in context.

    Returns (z, mask, meta, info) like loader.load_dem: z has NaN at NoData, mask is True
    where there is no bare earth, meta carries the grid. info is provenance: product,
    delivery years, tile keys, CRS, resolution, parcel coverage, and any mix and seam.
    """
    entries = entries if entries is not None else load_index()
    deliveries = deliveries_for(entries, aoi)
    if not deliveries:
        raise TileError("no LidarBC bare-earth (dem) tiles intersect this area in the "
                        f"indexed sheets ({', '.join(configured_sheets())})")
    choice = choose(deliveries)
    considered, best = [], None
    for d in [choice["primary"]] + choice["alternates"]:
        grid = plan_grid(aoi, d["tiles"][0]["crs"], d["tiles"][0]["res"])
        paths = _delivery_paths(d, aoi)
        z = assemble(paths, grid)
        pct = valid_pct(z, parcel_mask(parcel, grid))
        considered.append({"year": d["year"], "cover": d["cover"],
                           "parcel_valid_pct": round(pct, 1)})
        # newest first, so a later (older) delivery has to do strictly better
        if best is None or pct > best["pct"] + 1e-6:
            best = {"delivery": d, "grid": grid, "z": z, "paths": paths, "pct": pct}
        else:
            del z
        if pct >= MIN_PARCEL_VALID_PCT:
            break

    d, grid, z = best["delivery"], best["grid"], best["z"]
    used, paths = list(d["tiles"]), list(best["paths"])
    years = [d["year"]]
    mix = None
    for fill in choice["fill"]:
        fill_paths = _delivery_paths(fill, aoi)
        zf = assemble(fill_paths, grid)
        if not (np.isfinite(zf) & ~np.isfinite(z)).any():
            continue
        seam = seam_step(z, zf, grid["res"])
        if seam is None or abs(seam["step_m"]) > SEAM_MAX_STEP_M:
            mix = {
                "refused": True, "years": [d["year"], fill["year"]],
                "offset_m": None if seam is None else seam["step_m"],
                "reason": ("seam_unmeasurable" if seam is None else
                           f"seam step {seam['step_m']:+.2f} m between {d['year']} and "
                           f"{fill['year']} is over {SEAM_MAX_STEP_M} m; closed_hollow "
                           "reacts at 0.1 m, so a step reads as a dam or a hollow"),
            }
            break
        z = np.where(np.isfinite(z), z, zf).astype("float32")
        used.extend(fill["tiles"])
        paths.extend(fill_paths)
        years.append(fill["year"])
        mix = {"refused": False, "years": list(years), "offset_m": seam["step_m"],
               "method": seam["method"], "n": seam["n"]}
    mask = ~np.isfinite(z)
    meta = {
        "crs": grid["crs"], "transform": grid["transform"], "res": grid["res"],
        "bounds": grid["bounds"], "nodata": ND, "width": grid["width"],
        "height": grid["height"],
    }
    info = {
        "product": "dem",
        "years": years,
        "tile_keys": [t["key"] for t in used],
        "crs": str(grid["crs"]),
        "res_m": grid["res"],
        "aoi_resampled_m": grid["res"] if grid["res"] > 1.0 + 1e-6 else None,
        "deliveries_considered": considered,
        "parcel_valid_pct": round(valid_pct(z, parcel_mask(parcel, grid)), 1),
        "aoi_valid_pct": round(100.0 * float((~mask).mean()), 1),
        "mix": mix,
        "local_paths": paths,
    }
    return z, mask, meta, info


# ------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backend.pipeline.tiles",
                                 description="LidarBC tile index for the parcel path")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="list the bucket and read new DEM headers")
    r.add_argument("--sheet", action="append", help="sheet like 082e (repeatable)")
    r.add_argument("--workers", type=int, default=HEADER_WORKERS)
    f = sub.add_parser("find", help="deliveries covering a point and buffer")
    f.add_argument("--lat", type=float, required=True)
    f.add_argument("--lon", type=float, required=True)
    f.add_argument("--buffer", type=float, default=750.0)
    args = ap.parse_args(argv)

    if args.cmd == "refresh":
        for sheet in args.sheet or configured_sheets():
            refresh_index(sheet, workers=args.workers, log=print)
        return 0
    from shapely.geometry import Point
    xs, ys = transform("EPSG:4326", "EPSG:3005", [args.lon], [args.lat])
    aoi = Point(xs[0], ys[0]).buffer(args.buffer)
    for d in deliveries_for(load_index(), aoi):
        print(d["year"], f"cover {d['cover']:.3f}", [t["name"] for t in d["tiles"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
