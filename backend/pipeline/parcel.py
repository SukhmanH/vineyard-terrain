"""Resolve a listing to its parcel: PID, polygon and a confidence.

Inputs, highest priority first: a drawn geometry (GeoJSON or KML, WGS84 or EPSG:3005), a
CLSS lot id (confirmations only), PIDs, a point (lat/lon), an address and town. ParcelMap BC
parcels come from the DataBC WFS, because the geocoder's parcel resource is restricted to BC
ministries. Reserve lots come from NRCan's CLSS fabric (clss.py), because ParcelMap BC
carries each reserve as one Federal polygon.

What this module guarantees:
  - A drawn boundary that cannot be scored as given (unparseable, invalid, outside the area
    this service covers) raises BoundaryError and the job fails. The manual upload path's
    clip_to_boundary silently analyzes the whole tile instead, which for a listing would
    score the wrong ground.
  - Low confidence never scores. It returns candidates and reasons for the owner to confirm.
  - Parcels are never unioned silently. Several PIDs score as one unit only on an explicit
    confirmation, and only when they touch. A reserve is never scored as a whole.

Every field name read here was seen in a real response (docs/data-sources.md).
"""
import json
import re
import xml.etree.ElementTree as ET

from rasterio.warp import transform, transform_geom
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity

from . import clss, net

SQM_PER_ACRE = 4046.8564224
SOUTHERN_INTERIOR = (-120.6, 48.99, -118.8, 50.4)   # lon/lat box the parcel path serves
NEAR_SEARCH_M = 60.0        # a point on a road or sliver: candidates within this distance
ACRE_TOLERANCE = 0.15       # listing acres further off than this need confirmation
ADJACENT_M = 2.0            # parcels this close count as touching (multi-PID sales)
MAX_PARCEL_ACRES = 2000.0   # a bigger "parcel" (a reserve, a crown block) is never scored
MIN_BOUNDARY_ACRES = 0.05
GEOCODER_MIN_SCORE = 90
GEOCODER_PRECISION = "CIVIC_NUMBER"
NOT_FOR_SALE = ("Road", "Interest", "Park", "Return To Crown")   # PARCEL_CLASS values seen
WFS_COUNT = 200
PARCEL_LAYER = "pub:WHSE_CADASTRE.PMBC_PARCEL_POLY_SV"
PARCEL_FIELDS = ("PID,PID_FORMATTED,PARCEL_NAME,PLAN_NUMBER,PARCEL_CLASS,OWNER_TYPE,"
                 "PARCEL_STATUS,MUNICIPALITY,FEATURE_AREA_SQM,SHAPE")
PID_RE = re.compile(r"^\s*(\d{3})-?(\d{3})-?(\d{3})\s*$")


class BoundaryError(ValueError):
    """A drawn boundary that cannot be scored as given. Fails the job."""


class ResolveError(RuntimeError):
    """A lookup that could not be completed (service down, a page too many)."""


# --------------------------------------------------------------------- basics
def normalize_pid(pid):
    """'012525499' or '012-525-499' -> '012-525-499'. Anything else raises ValueError."""
    m = PID_RE.match(str(pid or ""))
    if not m:
        raise ValueError(f"not a PID (###-###-###): {pid!r}")
    return "-".join(m.groups())


def acres(geom):
    return geom.area / SQM_PER_ACRE


def to_wgs84(geom_3005):
    return transform_geom("EPSG:3005", "EPSG:4326", mapping(geom_3005))


def inside_service_area(geom_3005):
    lon0, lat0, lon1, lat1 = shape(to_wgs84(geom_3005)).bounds
    w, s, e, n = SOUTHERN_INTERIOR
    return lon0 >= w and lat0 >= s and lon1 <= e and lat1 <= n


def point_3005(lat, lon):
    xs, ys = transform("EPSG:4326", "EPSG:3005", [float(lon)], [float(lat)])
    return xs[0], ys[0]


def _acres_off(parcel_acres, listing_acres):
    return abs(parcel_acres - listing_acres) / listing_acres


# ------------------------------------------------------------- ParcelMap BC
def _wfs(cql):
    """Parcels matching a CQL filter, one bounded page.

    This layer cannot be paged: startIndex needs sortBy, and sortBy times out (504). A
    query matching more than one page is an error rather than a silently short list.
    """
    doc = net.get_json(net.WFS, {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": PARCEL_LAYER, "outputFormat": "application/json",
        "srsName": "EPSG:3005", "count": WFS_COUNT, "propertyName": PARCEL_FIELDS,
        "CQL_FILTER": cql,
    })
    feats = doc.get("features") or []
    matched = doc.get("numberMatched")
    if isinstance(matched, int) and matched > len(feats):
        raise ResolveError(f"ParcelMap BC matched {matched} parcels, more than one page "
                           f"({WFS_COUNT})")
    out = []
    for f in feats:
        p = f.get("properties") or {}
        if not f.get("geometry") or p.get("PARCEL_STATUS") not in (None, "Active"):
            continue
        g = shape(f["geometry"])
        out.append({
            "kind": "pmbc", "pid": p.get("PID_FORMATTED"), "parcel_name": p.get("PARCEL_NAME"),
            "plan": p.get("PLAN_NUMBER"), "parcel_class": p.get("PARCEL_CLASS"),
            "owner_type": p.get("OWNER_TYPE"), "municipality": p.get("MUNICIPALITY"),
            "acres": round(acres(g), 2), "geometry": g,
        })
    return out


def parcels_at(x, y):
    return _wfs(f"INTERSECTS(SHAPE,SRID=3005;POINT({x:.1f} {y:.1f}))")


def parcels_near(x, y, metres):
    return _wfs(f"DWITHIN(SHAPE,SRID=3005;POINT({x:.1f} {y:.1f}),{metres:g},meters)")


def parcels_in(bounds):
    return _wfs("BBOX(SHAPE,{:.1f},{:.1f},{:.1f},{:.1f},'EPSG:3005')".format(*bounds))


def parcels_by_pid(pid):
    """Parcels with this PID. Queries the indexed PID column: PID_FORMATTED is not indexed
    and the same lookup on it times out."""
    return _wfs(f"PID='{normalize_pid(pid).replace('-', '')}'")


def _for_sale(p):
    return p["parcel_class"] not in NOT_FOR_SALE


def _is_strata(p):
    return "strata" in (p["parcel_class"] or "").lower()


def _touching(parcel):
    zone = parcel["geometry"].buffer(ADJACENT_M)
    return [p for p in parcels_in(zone.bounds)
            if p["pid"] != parcel["pid"] and _for_sale(p) and p["geometry"].intersects(zone)]


# ------------------------------------------------------------------ geocoder
def geocode(address, town=None):
    """Best geocoder match for an address, with whether it is trusted to pick a parcel:
    a civic-number match, score >= GEOCODER_MIN_SCORE, and no LOCALITY fault."""
    query = ", ".join(s for s in (address, town, "BC") if s)
    feats = net.get_json(net.GEOCODER, {
        "addressString": query, "maxResults": 3, "outputSRS": 3005,
        "locationDescriptor": "parcelPoint",
    }).get("features") or []
    if not feats:
        return None
    f = feats[0]
    p = f.get("properties") or {}
    x, y = f["geometry"]["coordinates"][:2]
    faults = p.get("faults") or []
    locality_fault = any(str(x.get("element")) == "LOCALITY" for x in faults)
    return {
        "x": float(x), "y": float(y), "full_address": p.get("fullAddress"),
        "score": p.get("score"), "precision": p.get("matchPrecision"),
        "faults": [f"{x.get('element')}:{x.get('fault')}" for x in faults],
        "locality_fault": locality_fault,
        "ok": (p.get("matchPrecision") == GEOCODER_PRECISION
               and (p.get("score") or 0) >= GEOCODER_MIN_SCORE and not locality_fault),
    }


# ----------------------------------------------------------------- boundaries
def parse_boundary(boundary):
    """A drawn boundary (GeoJSON object or text, or KML text; WGS84 or EPSG:3005) as a
    polygon in EPSG:3005. Anything that cannot be scored as given raises BoundaryError."""
    if isinstance(boundary, str):
        text = boundary.strip()
        if text.startswith("<"):
            return _checked(_kml_polygon(text), "EPSG:4326")
        try:
            boundary = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BoundaryError(f"boundary is neither GeoJSON nor KML: {exc}") from exc
    if not isinstance(boundary, dict):
        raise BoundaryError("boundary must be GeoJSON (object or text) or KML text")
    geom, crs = _geojson_polygon(boundary)
    return _checked(geom, crs)


def _geojson_polygon(obj):
    kind = obj.get("type")
    if kind == "FeatureCollection":
        geoms = [f.get("geometry") for f in obj.get("features") or []]
    elif kind == "Feature":
        geoms = [obj.get("geometry")]
    else:
        geoms = [obj]
    shapes = []
    for g in geoms:
        if not isinstance(g, dict) or g.get("type") not in ("Polygon", "MultiPolygon"):
            got = g.get("type") if isinstance(g, dict) else type(g).__name__
            raise BoundaryError(f"boundary has a {got} where a polygon belongs")
        try:
            s = shape(g)
        except Exception as exc:
            raise BoundaryError(f"unreadable polygon: {exc}") from exc
        if not s.is_valid:
            raise BoundaryError(f"boundary is not a valid polygon ({explain_validity(s)})")
        shapes.append(s)
    if not shapes:
        raise BoundaryError("boundary has no polygon")
    name = str(((obj.get("crs") or {}).get("properties") or {}).get("name") or "")
    if "3005" in name:
        crs = "EPSG:3005"
    elif name and not any(k in name for k in ("4326", "CRS84")):
        raise BoundaryError(f"boundary CRS {name} is not WGS84 or EPSG:3005")
    else:
        crs = _guess_crs(unary_union(shapes).bounds)
    return (shapes[0] if len(shapes) == 1 else unary_union(shapes)), crs


def _guess_crs(bounds):
    minx, miny, maxx, maxy = bounds
    if -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90:
        return "EPSG:4326"
    if 200_000 <= minx and maxx <= 1_900_000 and 300_000 <= miny and maxy <= 1_800_000:
        return "EPSG:3005"
    raise BoundaryError("boundary coordinates are neither WGS84 degrees nor BC Albers "
                        "(EPSG:3005) metres")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _kml_ring(elem):
    for c in elem.iter():
        if _local(c.tag) == "coordinates" and c.text:
            return [tuple(float(v) for v in t.split(",")[:2]) for t in c.text.split()]
    return []


def _kml_polygon(text):
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise BoundaryError(f"unreadable KML: {exc}") from exc
    polys = []
    for poly in (e for e in root.iter() if _local(e.tag) == "Polygon"):
        outer, inners = [], []
        for child in poly:
            if _local(child.tag) == "outerBoundaryIs":
                outer = _kml_ring(child)
            elif _local(child.tag) == "innerBoundaryIs":
                inners.append(_kml_ring(child))
        if outer:
            polys.append({"type": "Polygon", "coordinates": [outer] + inners})
    if not polys:
        raise BoundaryError("KML has no Polygon")
    geom, _ = _geojson_polygon({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": g} for g in polys]})
    return geom


def _checked(geom, crs):
    if geom.is_empty:
        raise BoundaryError("boundary is empty")
    if not geom.is_valid:
        raise BoundaryError(f"boundary is not a valid polygon ({explain_validity(geom)})")
    if crs != "EPSG:3005":
        geom = shape(transform_geom(crs, "EPSG:3005", mapping(geom)))
    a = acres(geom)
    if not MIN_BOUNDARY_ACRES <= a <= MAX_PARCEL_ACRES:
        raise BoundaryError(f"boundary is {a:,.2f} ac; a parcel is {MIN_BOUNDARY_ACRES} to "
                            f"{MAX_PARCEL_ACRES:,.0f} ac")
    if not inside_service_area(geom):
        raise BoundaryError("boundary is outside the southern interior area this service "
                            "covers")
    return geom


# -------------------------------------------------------------------- results
def _resolved(geom, *, source, confidence, pids=(), owner_type=None, parcel_class=None,
              reasons=(), flags=(), point=None):
    return {
        "status": "resolved", "pid": pids[0] if len(pids) == 1 else None, "pids": list(pids),
        "geometry_3005": geom, "geometry_wgs84": to_wgs84(geom),
        "area_acres": round(acres(geom), 2), "owner_type": owner_type,
        "parcel_class": parcel_class, "source": source, "confidence": confidence,
        "candidates": [], "reasons": list(reasons), "flags": list(flags), "point_3005": point,
    }


def _candidate(i, c, point, listing_acres):
    pt = Point(point) if point else None
    return {
        "index": i, "kind": c.get("kind", "clss"), "pid": c.get("pid"),
        "clss_id": c.get("clss_id"), "designator": c.get("designator"),
        "acres": round(acres(c["geometry"]), 2), "owner_type": c.get("owner_type"),
        "parcel_class": c.get("parcel_class"),
        "distance_m": round(c["geometry"].distance(pt), 1) if pt else None,
        "acres_off_pct": (round(100 * _acres_off(acres(c["geometry"]), listing_acres))
                          if listing_acres else None),
        "geometry": c["geometry"],
    }


def _needs_confirmation(candidates, reasons, *, source, point=None, listing_acres=None):
    return {
        "status": "needs_confirmation", "pid": None, "pids": [], "geometry_3005": None,
        "geometry_wgs84": None, "area_acres": None, "owner_type": None, "parcel_class": None,
        "source": source, "confidence": "low",
        "candidates": [_candidate(i, c, point, listing_acres)
                       for i, c in enumerate(candidates, 1)],
        "reasons": list(reasons), "flags": [], "point_3005": point,
    }


def public(res):
    """The resolution without shapely geometries: JSON for the API and status files."""
    out = {k: v for k, v in res.items()
           if k not in ("geometry_3005", "geometry_wgs84", "point_3005", "candidates")}
    out["candidates"] = [{k: v for k, v in c.items() if k != "geometry"}
                         for c in res["candidates"]]
    return out


def feature_collection(res, aoi_3005=None):
    """parcel.geojson: the parcel, any candidates, and the AOI, in WGS84."""
    feats = []
    if res.get("geometry_3005") is not None:
        feats.append({"type": "Feature", "geometry": to_wgs84(res["geometry_3005"]),
                      "properties": {"role": "parcel", "pid": res.get("pid"),
                                     "pids": res.get("pids"), "acres": res.get("area_acres")}})
    for c in res.get("candidates") or []:
        feats.append({"type": "Feature", "geometry": to_wgs84(c["geometry"]),
                      "properties": {"role": "candidate", "index": c["index"],
                                     "pid": c.get("pid"), "clss_id": c.get("clss_id"),
                                     "designator": c.get("designator"),
                                     "acres": c["acres"]}})
    if aoi_3005 is not None:
        feats.append({"type": "Feature", "geometry": to_wgs84(aoi_3005),
                      "properties": {"role": "aoi"}})
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------- resolution
def resolve(req):
    """Resolve a parcel request: a dict with any of geometry, clss_id, pid (one or a list),
    lat/lon, address/town, plus listing_acres and confirmed. See the module docstring."""
    listing_acres = req.get("listing_acres")
    listing_acres = float(listing_acres) if listing_acres else None
    confirmed = bool(req.get("confirmed"))

    if req.get("geometry") is not None:
        return _from_geometry(parse_boundary(req["geometry"]))
    if req.get("clss_id"):
        if not confirmed:
            raise ValueError("a CLSS lot is only scored with confirmed: true")
        return _from_clss(req["clss_id"])
    if req.get("pid"):
        pids = req["pid"]
        pids = [p for p in re.split(r"[,\s]+", pids) if p] if isinstance(pids, str) else pids
        return _from_pids(pids, listing_acres, confirmed)
    if req.get("lat") is not None and req.get("lon") is not None:
        x, y = point_3005(req["lat"], req["lon"])
        return _from_point(x, y, listing_acres, confirmed, source="point",
                           confidence="medium")
    if req.get("address"):
        return _from_address(req["address"], req.get("town"), listing_acres, confirmed)
    raise ValueError("a parcel needs a geometry, PID, lat/lon or address")


def _gated(parcel, listing_acres, confirmed, *, source, confidence, point=None):
    """The resolved parcel, unless something about it needs the owner first."""
    if parcel["owner_type"] == "Federal" or parcel["acres"] > MAX_PARCEL_ACRES:
        # a PID names the whole reserve, so only a point says which lot is meant
        return _reserve_or_block(parcel, source, None if source == "pid" else point)
    reasons = []
    off = _acres_off(parcel["acres"], listing_acres) if listing_acres else None
    if off is not None and off > ACRE_TOLERANCE:
        label = f"parcel {parcel['pid']}" if parcel["pid"] else "the parcel"
        reasons.append(f"{label} is {parcel['acres']:.2f} ac but the listing says "
                       f"{listing_acres:g} ac ({off:.0%} off)")
    if _is_strata(parcel):
        reasons.append(f"{parcel['parcel_class']} parcel: strata lots are not individually "
                       "mapped in ParcelMap BC")
    if reasons and not confirmed:
        candidates = [parcel]
        if off is not None and listing_acres > parcel["acres"] * (1 + ACRE_TOLERANCE):
            candidates += _touching(parcel)
            reasons.append("the listing is larger than the parcel: it may be a multi-PID "
                           "sale (adjacent parcels listed)")
        return _needs_confirmation(candidates, reasons, source=source, point=point,
                                   listing_acres=listing_acres)
    flags = ["owner_confirmed"] if reasons else []
    return _resolved(parcel["geometry"], source=source, confidence=confidence,
                     pids=[parcel["pid"]] if parcel["pid"] else [],
                     owner_type=parcel["owner_type"], parcel_class=parcel["parcel_class"],
                     reasons=reasons, flags=flags, point=point)


def _reserve_or_block(parcel, source, point):
    """A reserve (or any parcel too big to be a listing) is never scored whole: offer the
    CLSS lots under the point, or ask for a drawn boundary."""
    if parcel["owner_type"] == "Federal":
        reasons = ["Federal land (a reserve): ParcelMap BC has no lots here"]
    else:
        reasons = [f"parcel is {parcel['acres']:,.0f} ac, too large to score as one listing"]
    found = clss.find_lots(Point(point)) if point else []
    lots = [lot for lot in found if acres(lot["geometry"]) <= MAX_PARCEL_ACRES]
    if lots:
        reasons.append("CLSS reserve lots at this point are listed")
    elif found:
        biggest = max(acres(lot["geometry"]) for lot in found)
        reasons.append(f"the CLSS lot here is {biggest:,.0f} ac, the unsubdivided remainder "
                       "of the reserve: a drawn boundary is required")
    elif point:
        reasons.append("no CLSS lot found here: a drawn boundary is required")
    else:
        reasons.append("send a location pin on the lot, or a drawn boundary")
    return _needs_confirmation(lots, reasons, source=source, point=point)


def _from_geometry(geom):
    """A drawn boundary is authoritative. ParcelMap BC is only asked which PID it is."""
    pids, notes = [], []
    try:
        for p in parcels_in(geom.bounds):
            if _for_sale(p) and p["geometry"].intersection(geom).area >= 0.9 * geom.area:
                pids = [p["pid"]] if p["pid"] else []
                break
    except (net.NetError, ResolveError) as exc:
        notes.append(f"ParcelMap BC lookup skipped: {exc}")
    return _resolved(geom, source="geometry", confidence="high", pids=pids, reasons=notes,
                     flags=["drawn_boundary"])


def _from_clss(clss_id):
    lot = clss.lot_by_id(clss_id)
    if lot is None:
        raise ValueError(f"no CLSS lot {clss_id!r}")
    if acres(lot["geometry"]) > MAX_PARCEL_ACRES:
        raise ValueError(f"CLSS lot {clss_id} is {acres(lot['geometry']):,.0f} ac, too large "
                         "to score as one listing; send a drawn boundary")
    return _resolved(lot["geometry"], source="clss", confidence="high", owner_type="Federal",
                     parcel_class=f"CLSS {lot['designator']}", flags=["reserve_lot"],
                     point=None)


def _from_pids(pids, listing_acres, confirmed):
    pids = list(dict.fromkeys(normalize_pid(p) for p in pids))
    if len(pids) > 1 and not confirmed:
        raise ValueError("several PIDs score as one unit only with confirmed: true")
    parcels = []
    for pid in pids:
        hits = parcels_by_pid(pid)
        if not hits:
            return _needs_confirmation([], [f"PID {pid} is not in ParcelMap BC"], source="pid")
        geom = unary_union([h["geometry"] for h in hits])
        parcels.append(dict(hits[0], geometry=geom, acres=round(acres(geom), 2)))
    if len(parcels) == 1:
        p = parcels[0]
        return _gated(p, listing_acres, confirmed, source="pid", confidence="high",
                      point=tuple(p["geometry"].representative_point().coords[0]))
    if any(p["owner_type"] == "Federal" for p in parcels):
        return _needs_confirmation([], ["a reserve parcel cannot be part of a multi-PID unit"],
                                   source="pid")
    if unary_union([p["geometry"].buffer(ADJACENT_M) for p in parcels]).geom_type != "Polygon":
        raise ValueError("these PIDs do not touch; a multi-PID listing is one block of "
                         "adjacent parcels")
    union = unary_union([p["geometry"] for p in parcels])
    if acres(union) > MAX_PARCEL_ACRES:
        raise ValueError(f"{acres(union):,.0f} ac is too large to score as one listing")
    classes = sorted({p["parcel_class"] for p in parcels if p["parcel_class"]})
    owners = sorted({p["owner_type"] for p in parcels if p["owner_type"]})
    return _resolved(union, source="pid", confidence="high", pids=[p["pid"] for p in parcels],
                     owner_type=", ".join(owners) or None,
                     parcel_class=", ".join(classes) or None, flags=["multi_pid"])


def _from_point(x, y, listing_acres, confirmed, *, source, confidence):
    point = (x, y)
    at = parcels_at(x, y)
    sale = [p for p in at if _for_sale(p)]
    if len(sale) == 1:
        return _gated(sale[0], listing_acres, confirmed, source=source,
                      confidence=confidence, point=point)
    if len(sale) > 1:
        return _needs_confirmation(sale, [f"{len(sale)} parcels overlap at this point"],
                                   source=source, point=point, listing_acres=listing_acres)
    near = [p for p in parcels_near(x, y, NEAR_SEARCH_M) if _for_sale(p)]
    why = ("the point is on a road or right of way" if at else
           "the point is not inside any parcel")
    why += f"; {len(near)} parcels within {NEAR_SEARCH_M:g} m"
    if listing_acres and near:
        fits = [p for p in near if _acres_off(p["acres"], listing_acres) <= ACRE_TOLERANCE]
        if len(fits) == 1:
            res = _gated(fits[0], listing_acres, confirmed, source=source,
                         confidence="medium", point=point)
            res["reasons"].insert(0, why + "; exactly one matches the listing acres")
            return res
    pt = Point(point)
    near.sort(key=lambda p: (_acres_off(p["acres"], listing_acres) if listing_acres else 0.0,
                             p["geometry"].distance(pt)))
    return _needs_confirmation(near, [why], source=source, point=point,
                               listing_acres=listing_acres)


def _from_address(address, town, listing_acres, confirmed):
    g = geocode(address, town)
    if g is None:
        return _needs_confirmation([], [f"the geocoder found no match for {address!r}"],
                                   source="address")
    if not g["ok"]:
        why = (f"address matched {g['full_address']!r} at {g['precision']} precision, "
               f"score {g['score']}")
        if g["faults"]:
            why += f" (faults: {', '.join(g['faults'])})"
        if g["locality_fault"] or g["precision"] not in ("BLOCK", "STREET"):
            return _needs_confirmation([], [why, "not trusted to pick a parcel: send a PID "
                                            "or a location pin"], source="address")
        near = [p for p in parcels_near(g["x"], g["y"], NEAR_SEARCH_M) if _for_sale(p)]
        return _needs_confirmation(near, [why], source="address", point=(g["x"], g["y"]),
                                   listing_acres=listing_acres)
    res = _from_point(g["x"], g["y"], listing_acres, confirmed, source="address",
                      confidence="high")
    res["geocode"] = {k: g[k] for k in ("full_address", "score", "precision", "faults")}
    return res
