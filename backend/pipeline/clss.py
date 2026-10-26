"""Canada Lands (CLSS) lots for reserve parcels that ParcelMap BC does not split up.

ParcelMap BC carries each reserve as one Federal polygon (Osoyoos IR 1 is a single 32,069 ac
"Part of Primary" parcel), so band leasehold lots and the owner's OIB/PIB leases are not in
it. NRCan publishes each reserve's survey fabric as a shapefile. It is downloaded once, kept
in data/clss/ with its licence notice, and its lots are read straight from the zip (a
fraction of a second), so there is no derived copy to keep in step with the download.

The reader is a small stdlib reader for the parts of the shapefile format these files use
(Polygon, PolygonZ and PolygonM records, a dBase III attribute table, a WKT .prj), which
keeps a one-off import from adding a dependency.

    python -m backend.pipeline.clss fetch      # download the zips (needs the network once)
    python -m backend.pipeline.clss summary    # layers and lot counts in the zips
"""
import argparse
import functools
import io
import os
import struct
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLSS_DIR = os.path.join(ROOT, "data", "clss")
FTP_HOST = "ftp.nrcan.gc.ca"
FTP_BASE = f"ftp://{FTP_HOST}/ess/sgb_pub/sgb_datasets/bc/"
# NRCan folder names, which are not the reserves' short names (there is no OSOYOOS_1)
RESERVES = ("OSOYOOS_INDIAN_RESERVE_NO_1", "PENTICTON_1")
LOT_LAYER = "Land_Parcel"       # the active lot fabric; the zips also carry retired lots,
                                # easements, plan envelopes and surveys in progress
ID_FIELD = "PIN"                # unique per lot in both files (925 and 1,220 lots)
NOT_A_LOT = ("ROAD R\\W", "LANE", "WALKWAY")   # DESIGNATOR values that are never for sale
SQM_PER_ACRE = 4046.8564224
POLYGON_TYPES = (5, 15, 25)     # Polygon, PolygonZ, PolygonM


class ShapefileError(ValueError):
    pass


# ------------------------------------------------------------------- reading
def read_shp(data):
    """Polygon geometries from .shp bytes, in record order (None for null shapes)."""
    if len(data) < 100 or struct.unpack(">i", data[:4])[0] != 9994:
        raise ShapefileError("not a shapefile (.shp)")
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type not in POLYGON_TYPES:
        raise ShapefileError(f"shape type {shape_type} is not a polygon layer")
    out, pos = [], 100
    while pos + 8 <= len(data):
        _, words = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + 2 * words]
        pos += 8 + 2 * words
        if struct.unpack("<i", content[:4])[0] == 0:
            out.append(None)
            continue
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts])
        off = 44 + 4 * n_parts
        xy = struct.unpack(f"<{2 * n_points}d", content[off:off + 16 * n_points])
        pts = list(zip(xy[0::2], xy[1::2]))
        ends = list(parts) + [n_points]
        out.append(_rings_to_geometry([pts[ends[i]:ends[i + 1]] for i in range(n_parts)]))
    return out


def _signed_area(ring):
    return 0.5 * sum(x0 * y1 - x1 * y0
                     for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]))


def _rings_to_geometry(rings):
    """Shapefile polygons: clockwise rings are shells, counter-clockwise rings are holes."""
    shells, holes = [], []
    for r in rings:
        if len(r) >= 4:
            (shells if _signed_area(r) < 0 else holes).append(r)
    if not shells:                       # some writers ignore the winding rule
        shells, holes = holes, []
    polys = [[s, []] for s in shells]
    for h in holes:
        probe = Point(h[0])
        for p in polys:
            if Polygon(p[0]).contains(probe):
                p[1].append(h)
                break
    geoms = [Polygon(s, hs) for s, hs in polys]
    return geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)


def read_dbf(data, encoding="latin-1"):
    """Records of a dBase III table as dicts of stripped strings (None if deleted)."""
    n_records, header_len, record_len = struct.unpack("<IHH", data[4:12])
    fields, pos = [], 32
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b"\x00")[0].decode("ascii", "replace")
        fields.append((name, data[pos + 16]))
        pos += 32
    out = []
    for i in range(n_records):
        rec = data[header_len + i * record_len:header_len + (i + 1) * record_len]
        if not rec or rec[:1] == b"*":
            out.append(None)
            continue
        row, off = {}, 1
        for name, length in fields:
            row[name] = rec[off:off + length].decode(encoding, "replace").strip()
            off += length
        out.append(row)
    return out


def layers_in_zip(zip_bytes):
    """Shapefile layers in a zip: {stem: {"shp": bytes, "dbf": bytes, "prj": bytes, ...}}."""
    layers = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            base, ext = os.path.splitext(info.filename)
            ext = ext.lower().lstrip(".")
            if ext in ("shp", "dbf", "prj", "cpg"):
                layers.setdefault(base, {})[ext] = zf.read(info)
    return {k: v for k, v in layers.items() if "shp" in v and "dbf" in v}


def read_layer(parts):
    """GeoJSON-like features (EPSG:3005) of one polygon layer."""
    if not parts.get("prj"):
        raise ShapefileError("layer has no .prj; refusing to guess its CRS")
    src = CRS.from_wkt(parts["prj"].decode("latin-1"))
    cpg = (parts.get("cpg") or b"").decode("ascii", "replace").lower()
    rows = read_dbf(parts["dbf"], "utf-8" if "utf" in cpg else "latin-1")
    feats = []
    for geom, row in zip(read_shp(parts["shp"]), rows):
        if geom is None or row is None:
            continue
        feats.append({"type": "Feature", "properties": row,
                      "geometry": transform_geom(src, "EPSG:3005", mapping(geom))})
    return feats


# ------------------------------------------------------------------- lookups
def zip_path(reserve):
    return os.path.join(CLSS_DIR, reserve, f"{reserve}_SHP.zip")


@functools.lru_cache(maxsize=1)
def _all_lots():
    lots = []
    for reserve in RESERVES:
        path = zip_path(reserve)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            layer = layers_in_zip(f.read()).get(LOT_LAYER)
        if layer is None:
            continue
        for feat in read_layer(layer):
            props = feat["properties"]
            lots.append({
                "reserve": reserve,
                "clss_id": f"{reserve}:{props.get(ID_FIELD)}",
                "designator": props.get("DESIGNATOR"),
                "plan": props.get("PLAN_NUM"),
                "properties": props,
                "geometry": shape(feat["geometry"]),
            })
    return tuple(lots)


def load_lots():
    """Every lot (shapely, EPSG:3005) in the downloaded reserve fabrics, roads excluded."""
    return [lot for lot in _all_lots() if lot["designator"] not in NOT_A_LOT]


def find_lots(point_3005, within_m=0.0, lots=None):
    """Lots containing a point (EPSG:3005), or within within_m of it."""
    lots = load_lots() if lots is None else lots
    probe = point_3005.buffer(within_m) if within_m else point_3005
    return [lot for lot in lots if lot["geometry"].intersects(probe)]


def lot_by_id(clss_id, lots=None):
    lots = load_lots() if lots is None else lots
    return next((lot for lot in lots if lot["clss_id"] == clss_id), None)


# ------------------------------------------------------------------- the CLI
def _ftp_get(url):
    if urllib.parse.urlsplit(url).hostname != FTP_HOST:
        raise ValueError(f"CLSS downloads only come from {FTP_HOST}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def listing_date(line, today=None):
    """ISO date of an FTP LIST line. Unix listings omit the year for files under six months
    old and show the time of day instead."""
    month, day, year_or_time = line.split()[5:8]
    today = today or datetime.now(timezone.utc).date()
    if ":" in year_or_time:
        when = datetime.strptime(f"{month} {day} {today.year}", "%b %d %Y").date()
        if when > today:
            when = when.replace(year=today.year - 1)
    else:
        when = datetime.strptime(f"{month} {day} {year_or_time}", "%b %d %Y").date()
    return when.isoformat()


def fetch():
    """Download each reserve's SHP zip once into data/clss/<RESERVE>/ with a SOURCE.md."""
    for reserve in RESERVES:
        folder = os.path.join(CLSS_DIR, reserve)
        os.makedirs(folder, exist_ok=True)
        name = f"{reserve}_SHP.zip"
        url = f"{FTP_BASE}{reserve}/{name}"
        listing = _ftp_get(f"{FTP_BASE}{reserve}/").decode("latin-1")
        line = next((ln for ln in listing.splitlines() if ln.endswith(" " + name)), "")
        data = _ftp_get(url)
        with open(zip_path(reserve), "wb") as f:
            f.write(data)
        dated = listing_date(line) if line else "unknown"
        lines = [
            f"# {reserve}", "",
            f"- Source: {url}",
            f"- Size: {len(data):,} bytes",
            f"- Dataset date (FTP listing): {dated}",
            f"- Retrieved: {datetime.now(timezone.utc):%Y-%m-%d}",
            f"- Lots: layer `{LOT_LAYER}`, id `{ID_FIELD}`",
            "- Licence: Open Government Licence - Canada (see ../README.md)",
        ]
        with open(os.path.join(folder, "SOURCE.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{reserve}: {len(data):,} bytes, dated {dated}")


def summary():
    lots = _all_lots()
    for reserve in RESERVES:
        if not os.path.exists(zip_path(reserve)):
            print(f"{reserve}: not downloaded")
            continue
        with open(zip_path(reserve), "rb") as f:
            layers = layers_in_zip(f.read())
        mine = [lot for lot in lots if lot["reserve"] == reserve]
        acres = sum(lot["geometry"].area for lot in mine) / SQM_PER_ACRE
        print(f"{reserve}: layers {sorted(layers)}; {LOT_LAYER}: {len(mine)} lots, "
              f"{acres:,.0f} ac")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backend.pipeline.clss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("summary")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        fetch()
    else:
        summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
