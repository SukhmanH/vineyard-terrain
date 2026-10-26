"""Parcel resolution (backend/pipeline/parcel.py) and the CLSS reader (clss.py).

Every WFS and geocoder response here is a trimmed real response in tests/fixtures/ (see its
README): 1056 Naramata Rd, the Cassini coordinates that land in no parcel, the Cassini
address that the geocoder places in Peachland, the Osoyoos reserve polygon, and samples of
strata and right-of-way parcels.
"""
import io
import json
import os
import struct
import zipfile

import pytest
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import Point, box, mapping, shape

from backend.pipeline import clss, net, parcel

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
NARAMATA_1056 = (1465742.0, 522753.0)            # geocoder parcelPoint, EPSG:3005
CASSINI = (49.134041, -119.582336)               # the owner's coordinates for Cassini
BLACK_SAGE = (49.150, -119.505)                  # inside Osoyoos IR 1


def load(rel):
    with open(os.path.join(FIX, rel), encoding="utf-8") as f:
        return json.load(f)


def fc(features, matched=None):
    return {"type": "FeatureCollection", "features": features,
            "numberMatched": len(features) if matched is None else matched}


def by_pid(*fixtures):
    """A WFS route answering PID='#########' from the parcels in some fixtures."""
    feats = [f for fx in fixtures for f in load(fx)["features"]]

    def answer(cql):
        digits = cql.split("'")[1]
        return fc([f for f in feats if (f["properties"]["PID"] or "") == digits])
    return answer


class Services:
    """Stand-in for the WFS and geocoder: each request must match a recorded route."""

    def __init__(self, monkeypatch):
        self.wfs, self.geo, self.calls = [], [], []
        monkeypatch.setattr(net, "get_json", self)

    def __call__(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        if url == net.GEOCODER:
            for needle, fx in self.geo:
                if needle in params["addressString"]:
                    return load(fx)
        elif url == net.WFS:
            cql = params["CQL_FILTER"]
            for needle, fx in self.wfs:
                if needle in cql:
                    return fx(cql) if callable(fx) else (load(fx) if isinstance(fx, str) else fx)
        raise AssertionError(f"unexpected request {url} {params}")

    def cql(self):
        return [p["CQL_FILTER"] for u, p in self.calls if u == net.WFS]


@pytest.fixture
def svc(monkeypatch):
    return Services(monkeypatch)


def _naramata(svc):
    svc.geo.append(("1056 Naramata", "geocoder/naramata_1056.json"))
    svc.wfs.append(("INTERSECTS(SHAPE,SRID=3005;POINT(1465742.0 522753.0))",
                    "wfs/at_naramata_1056.json"))
    svc.wfs.append(("BBOX(SHAPE", "wfs/bbox_naramata_1056.json"))
    svc.wfs.append(("PID='", by_pid("wfs/bbox_naramata_1056.json", "wfs/near_cassini.json")))


def _cassini_point(svc):
    x, y = parcel.point_3005(*CASSINI)
    svc.wfs.append((f"INTERSECTS(SHAPE,SRID=3005;POINT({x:.1f} {y:.1f}))", "wfs/at_cassini.json"))
    svc.wfs.append((f"DWITHIN(SHAPE,SRID=3005;POINT({x:.1f} {y:.1f}),60,meters)",
                    "wfs/near_cassini.json"))


# ------------------------------------------------------------------ address
def test_a_civic_address_resolves_to_the_parcel_under_its_parcel_point(svc):
    _naramata(svc)
    res = parcel.resolve({"address": "1056 Naramata Rd", "town": "Penticton"})
    assert res["status"] == "resolved" and res["confidence"] == "high"
    assert (res["pid"], res["source"]) == ("012-525-499", "address")
    assert res["area_acres"] == pytest.approx(9.0, abs=0.1)
    assert res["owner_type"] == "Private" and res["parcel_class"] == "Subdivision"
    assert res["geocode"]["precision"] == "CIVIC_NUMBER"
    assert res["geometry_wgs84"]["type"] in ("Polygon", "MultiPolygon")


def test_listing_acres_within_tolerance_still_resolves(svc):
    _naramata(svc)
    res = parcel.resolve({"address": "1056 Naramata Rd", "town": "Penticton",
                          "listing_acres": 9.8})
    assert res["status"] == "resolved"


def test_a_listing_much_larger_than_the_parcel_offers_adjacent_parcels(svc):
    _naramata(svc)
    res = parcel.resolve({"address": "1056 Naramata Rd", "town": "Penticton",
                          "listing_acres": 20})
    assert res["status"] == "needs_confirmation" and res["confidence"] == "low"
    assert res["candidates"][0]["pid"] == "012-525-499"
    assert len(res["candidates"]) > 1                  # its neighbours, never a silent union
    assert any("multi-PID" in r for r in res["reasons"])
    assert res["geometry_3005"] is None


def test_an_address_the_geocoder_puts_in_another_town_is_not_trusted(svc):
    """'4798 Okanagan Hwy, Oliver' comes back as 4798 Hwy 97, Peachland (BLOCK, score 87)."""
    svc.geo.append(("4798 Okanagan Hwy", "geocoder/cassini_peachland.json"))
    res = parcel.resolve({"address": "4798 Okanagan Hwy", "town": "Oliver"})
    assert res["status"] == "needs_confirmation"
    assert res["candidates"] == [] and res["point_3005"] is None   # no Peachland parcels
    assert "Peachland" in res["reasons"][0] and "BLOCK" in res["reasons"][0]
    assert svc.cql() == []


def test_geocoder_trust_rules_on_real_matches(svc):
    svc.geo += [("70 Upper Bench", "geocoder/upper_bench_direction.json"),
                ("Keremeos", "geocoder/locality_only.json"),
                ("1056 Naramata", "geocoder/naramata_1056.json")]
    upper = parcel.geocode("70 Upper Bench Rd", "Penticton")
    assert upper["ok"] and "STREET_DIRECTION:missing" in upper["faults"]   # score 98 is fine
    assert parcel.geocode("Keremeos")["ok"] is False                      # locality-level
    good = parcel.geocode("1056 Naramata Rd", "Penticton")
    assert good["ok"] and (good["x"], good["y"]) == NARAMATA_1056


# -------------------------------------------------------------------- point
def test_a_point_in_no_parcel_returns_the_parcels_within_60_m(svc):
    _cassini_point(svc)
    res = parcel.resolve({"lat": CASSINI[0], "lon": CASSINI[1]})
    assert res["status"] == "needs_confirmation"
    assert {c["pid"] for c in res["candidates"]} == {
        "011-333-600", "011-333-596", "008-646-074", "011-333-570"}
    dist = [c["distance_m"] for c in res["candidates"]]
    assert dist == sorted(dist) and dist[0] < 60
    assert "not inside any parcel" in res["reasons"][0]


def test_a_unique_acreage_match_near_the_point_resolves_at_medium(svc):
    _cassini_point(svc)
    res = parcel.resolve({"lat": CASSINI[0], "lon": CASSINI[1], "listing_acres": 14})
    assert res["status"] == "resolved" and res["confidence"] == "medium"
    assert res["pid"] == "008-646-074"
    two_fit = parcel.resolve({"lat": CASSINI[0], "lon": CASSINI[1], "listing_acres": 9.7})
    assert two_fit["status"] == "needs_confirmation"     # 10.23 and 9.2 ac both fit
    assert [c["pid"] for c in two_fit["candidates"]][:2] == ["011-333-596", "011-333-600"]


def test_right_of_way_parcels_are_never_candidates(svc):
    roads = load("wfs/road_interest_naramata.json")["features"]
    assert {f["properties"]["PARCEL_CLASS"] for f in roads} == {"Interest"}
    x, y = shape(roads[0]["geometry"]).representative_point().coords[0]
    lon, lat = [v[0] for v in transform("EPSG:3005", "EPSG:4326", [x], [y])]
    svc.wfs.append(("INTERSECTS", fc(roads[:1])))
    svc.wfs.append(("DWITHIN", fc(roads)))
    res = parcel.resolve({"lat": lat, "lon": lon})
    assert res["status"] == "needs_confirmation" and res["candidates"] == []
    assert "road or right of way" in res["reasons"][0]


def test_a_strata_parcel_needs_confirmation(svc):
    strata = load("wfs/strata_penticton.json")["features"][:1]
    assert strata[0]["properties"]["PID_FORMATTED"] is None   # strata lots are not mapped
    x, y = shape(strata[0]["geometry"]).representative_point().coords[0]
    lon, lat = [v[0] for v in transform("EPSG:3005", "EPSG:4326", [x], [y])]
    svc.wfs.append(("INTERSECTS", fc(strata)))
    res = parcel.resolve({"lat": lat, "lon": lon})
    assert res["status"] == "needs_confirmation"
    assert "Building Strata" in res["reasons"][0]
    ok = parcel.resolve({"lat": lat, "lon": lon, "confirmed": True})
    assert ok["status"] == "resolved" and "owner_confirmed" in ok["flags"]


# ------------------------------------------------------------------ reserve
def test_reserve_land_offers_clss_lots_and_is_never_scored_whole(svc, monkeypatch):
    svc.wfs.append(("INTERSECTS", "wfs/federal_black_sage.json"))
    x, y = parcel.point_3005(*BLACK_SAGE)
    lot = {"reserve": "OSOYOOS_INDIAN_RESERVE_NO_1", "clss_id": "OSOYOOS_INDIAN_RESERVE_NO_1:1",
           "designator": "LOT 1", "geometry": Point(x, y).buffer(100)}
    monkeypatch.setattr(clss, "find_lots", lambda pt, within_m=0.0, lots=None: [lot])
    for confirmed in (False, True):
        res = parcel.resolve({"lat": BLACK_SAGE[0], "lon": BLACK_SAGE[1],
                              "confirmed": confirmed})
        assert res["status"] == "needs_confirmation"
        assert res["reasons"][0].startswith("Federal land")
        assert [c["clss_id"] for c in res["candidates"]] == [lot["clss_id"]]
    monkeypatch.setattr(clss, "find_lots", lambda pt, within_m=0.0, lots=None: [])
    none = parcel.resolve({"lat": BLACK_SAGE[0], "lon": BLACK_SAGE[1]})
    assert "drawn boundary is required" in none["reasons"][-1]
    # the real lot under this point is the 26,076 ac remainder of the reserve
    remainder = dict(lot, geometry=Point(x, y).buffer(5800))
    monkeypatch.setattr(clss, "find_lots", lambda pt, within_m=0.0, lots=None: [remainder])
    big = parcel.resolve({"lat": BLACK_SAGE[0], "lon": BLACK_SAGE[1]})
    assert big["candidates"] == [] and "unsubdivided remainder" in big["reasons"][-1]


def test_a_confirmed_clss_lot_resolves(monkeypatch):
    lot = {"clss_id": "PENTICTON_1:1207513", "designator": "LOT 455-1",
           "geometry": box(1462000, 515000, 1462200, 515200)}
    monkeypatch.setattr(clss, "lot_by_id", lambda cid, lots=None: lot if cid == lot["clss_id"]
                        else None)
    with pytest.raises(ValueError, match="confirmed"):
        parcel.resolve({"clss_id": lot["clss_id"]})
    res = parcel.resolve({"clss_id": lot["clss_id"], "confirmed": True})
    assert res["status"] == "resolved" and res["source"] == "clss"
    assert res["area_acres"] == pytest.approx(40000 / parcel.SQM_PER_ACRE, abs=0.01)


# --------------------------------------------------------------------- PIDs
def test_a_pid_is_looked_up_on_the_indexed_column(svc):
    _naramata(svc)
    res = parcel.resolve({"pid": "012525499"})
    assert res["status"] == "resolved" and res["pid"] == "012-525-499"
    assert res["source"] == "pid" and res["confidence"] == "high"
    assert svc.cql() == ["PID='012525499'"]          # PID_FORMATTED is not indexed


def test_malformed_pids_are_refused():
    for bad in ("12-525-499", "012525499x", "DROP TABLE", ""):
        with pytest.raises(ValueError):
            parcel.normalize_pid(bad)


def _touching_neighbour():
    feats = load("wfs/bbox_naramata_1056.json")["features"]
    me = next(shape(f["geometry"]) for f in feats
              if f["properties"]["PID_FORMATTED"] == "012-525-499")
    return next(f["properties"]["PID_FORMATTED"] for f in feats
                if f["properties"]["PID_FORMATTED"] not in (None, "012-525-499")
                and shape(f["geometry"]).distance(me) < 1.0)


def test_several_pids_score_as_one_unit_only_when_confirmed_and_touching(svc):
    _naramata(svc)
    neighbour = _touching_neighbour()
    with pytest.raises(ValueError, match="confirmed"):
        parcel.resolve({"pid": ["012-525-499", neighbour]})
    res = parcel.resolve({"pid": f"012-525-499,{neighbour}", "confirmed": True})
    assert res["status"] == "resolved" and "multi_pid" in res["flags"]
    assert res["pids"] == ["012-525-499", neighbour] and res["pid"] is None
    assert res["geometry_3005"].geom_type == "Polygon"
    with pytest.raises(ValueError, match="do not touch"):
        parcel.resolve({"pid": ["012-525-499", "011-333-600"], "confirmed": True})


def test_an_unknown_pid_needs_confirmation(svc):
    svc.wfs.append(("PID='", fc([])))
    res = parcel.resolve({"pid": "999-999-999"})
    assert res["status"] == "needs_confirmation" and "not in ParcelMap BC" in res["reasons"][0]


def test_a_query_matching_more_than_one_page_is_an_error(svc):
    svc.wfs.append(("BBOX", fc([], matched=parcel.WFS_COUNT + 1)))
    with pytest.raises(parcel.ResolveError, match="more than one page"):
        parcel.parcels_in((0, 0, 1, 1))


# ---------------------------------------------------------------- boundaries
WGS_SQUARE = [[-119.5620, 49.5300], [-119.5600, 49.5300], [-119.5600, 49.5312],
              [-119.5620, 49.5312], [-119.5620, 49.5300]]


@pytest.fixture
def no_pmbc(monkeypatch):
    monkeypatch.setattr(parcel, "parcels_in", lambda bounds: [])


def test_a_drawn_geojson_boundary_is_authoritative(no_pmbc):
    feature = {"type": "Feature", "properties": {},
               "geometry": {"type": "Polygon", "coordinates": [WGS_SQUARE]}}
    res = parcel.resolve({"geometry": feature, "listing_acres": 99})
    assert res["status"] == "resolved" and res["source"] == "geometry"
    assert res["confidence"] == "high" and "drawn_boundary" in res["flags"]
    assert res["area_acres"] == pytest.approx(4.77, abs=0.05)    # 145 m x 133 m


def test_boundaries_in_bc_albers_and_kml_parse(no_pmbc):
    albers = {"type": "Polygon", "coordinates": [[[1465600, 522600], [1465800, 522600],
                                                  [1465800, 522800], [1465600, 522800],
                                                  [1465600, 522600]]]}
    assert parcel.parse_boundary(albers).area == pytest.approx(40000)
    named = dict(albers, crs={"type": "name", "properties": {"name": "EPSG:3005"}})
    assert parcel.parse_boundary(json.dumps(named)).area == pytest.approx(40000)
    coords = " ".join(f"{x},{y},0" for x, y in WGS_SQUARE)
    kml = ('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Placemark>'
           f'<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates>'
           '</LinearRing></outerBoundaryIs></Polygon></Placemark></kml>')
    from_kml = parcel.parse_boundary(kml)
    from_json = parcel.parse_boundary({"type": "Polygon", "coordinates": [WGS_SQUARE]})
    assert from_kml.symmetric_difference(from_json).area < 1.0


@pytest.mark.parametrize("bad, match", [
    ({"type": "Polygon", "coordinates": [[[-119.562, 49.53], [-119.560, 49.532],
                                          [-119.560, 49.530], [-119.562, 49.532],
                                          [-119.562, 49.53]]]}, "not a valid polygon"),
    ({"type": "Point", "coordinates": [-119.56, 49.53]}, "where a polygon belongs"),
    ({"type": "Polygon", "coordinates": [[[-123.12, 49.28], [-123.10, 49.28],
                                          [-123.10, 49.29], [-123.12, 49.29],
                                          [-123.12, 49.28]]]}, "outside"),
    ({"type": "Polygon", "coordinates": [[[-119.56, 49.53], [-119.5599, 49.53],
                                          [-119.5599, 49.5301], [-119.56, 49.5301],
                                          [-119.56, 49.53]]]}, "a parcel is"),
    ("this is not a boundary", "neither GeoJSON nor KML"),
    ("<kml><Placemark/></kml>", "no Polygon"),
    ({"type": "Polygon", "coordinates": [[[5e6, 5e6], [5e6 + 10, 5e6], [5e6, 5e6 + 10],
                                          [5e6, 5e6]]]}, "neither WGS84"),
])
def test_a_bad_boundary_fails_instead_of_scoring_other_ground(bad, match):
    with pytest.raises(parcel.BoundaryError, match=match):
        parcel.parse_boundary(bad)


def test_public_output_carries_no_geometry_and_geojson_does(svc):
    _cassini_point(svc)
    res = parcel.resolve({"lat": CASSINI[0], "lon": CASSINI[1]})
    pub = parcel.public(res)
    json.dumps(pub)                                   # JSON-safe
    assert "geometry_3005" not in pub and all("geometry" not in c for c in pub["candidates"])
    fcol = parcel.feature_collection(res, aoi_3005=Point(*res["point_3005"]).buffer(800))
    roles = [f["properties"]["role"] for f in fcol["features"]]
    assert roles.count("candidate") == 4 and roles[-1] == "aoi"


# --------------------------------------------------------------------- CLSS
def _ring(x0, y0, x1, y1, clockwise=True):
    r = [(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    return r if clockwise else r[::-1]


def _shp(polys):
    recs = b""
    for i, rings in enumerate(polys, 1):
        pts = [p for r in rings for p in r]
        parts, n = [], 0
        for r in rings:
            parts.append(n)
            n += len(r)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        body = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), len(rings),
                           len(pts))
        body += struct.pack(f"<{len(parts)}i", *parts)
        body += struct.pack(f"<{2 * len(pts)}d", *[c for p in pts for c in p])
        recs += struct.pack(">2i", i, len(body) // 2) + body
    head = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2)
    return head + struct.pack("<2i", 1000, 5) + struct.pack("<8d", *[0.0] * 8) + recs


def _dbf(rows, fields):
    head = struct.pack("<B3BIHH20x", 3, 126, 10, 26, len(rows), 33 + 32 * len(fields),
                       1 + sum(n for _, n in fields))
    for name, n in fields:
        head += struct.pack("<11sc4xBB14x", name.encode(), b"C", n, 0)
    body = b"".join(b" " + b"".join(str(r[k]).encode().ljust(n)[:n] for k, n in fields)
                    for r in rows)
    return head + b"\r" + body + b"\x1a"


def test_the_shapefile_reader_handles_holes_attributes_and_crs():
    outer = _ring(312000, 5452000, 312200, 5452100)
    hole = _ring(312050, 5452020, 312100, 5452070, clockwise=False)
    shp = _shp([[outer, hole], [_ring(312300, 5452000, 312400, 5452100)]])
    dbf = _dbf([{"PIN": "1251036", "DESIGNATOR": "LOT 395"},
                {"PIN": "1251037", "DESIGNATOR": "ROAD R\\W"}],
               [("PIN", 10), ("DESIGNATOR", 20)])
    geoms = clss.read_shp(shp)
    assert geoms[0].area == pytest.approx(200 * 100 - 50 * 50)       # the hole is a hole
    assert clss.read_dbf(dbf)[1] == {"PIN": "1251037", "DESIGNATOR": "ROAD R\\W"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Land_Parcel.shp", shp)
        zf.writestr("Land_Parcel.dbf", dbf)
        zf.writestr("Land_Parcel.prj", CRS.from_epsg(2955).to_wkt())
        zf.writestr("Land_Parcel.cpg", "UTF-8")
        zf.writestr("Ancillary_Line.shx", b"")
    layers = clss.layers_in_zip(buf.getvalue())
    assert list(layers) == ["Land_Parcel"]
    feats = clss.read_layer(layers["Land_Parcel"])
    first = shape(feats[0]["geometry"])
    assert first.bounds[0] > 1_000_000                               # now BC Albers metres
    assert first.area == pytest.approx(17500, rel=0.01)
    with pytest.raises(clss.ShapefileError, match="prj"):
        clss.read_layer({"shp": shp, "dbf": dbf})


def test_the_committed_reserve_fabrics_load():
    lots = clss.load_lots()
    by_reserve = {r: sum(1 for lot in lots if lot["reserve"] == r) for r in clss.RESERVES}
    assert by_reserve["OSOYOOS_INDIAN_RESERVE_NO_1"] > 850       # 925 less road lots
    assert by_reserve["PENTICTON_1"] > 1100                      # 1,220 less road lots
    assert len({lot["clss_id"] for lot in lots}) == len(lots)
    assert not any(lot["designator"] in clss.NOT_A_LOT for lot in lots)
    x, y = parcel.point_3005(*BLACK_SAGE)
    hits = clss.find_lots(Point(x, y), within_m=500)
    assert all(h["reserve"] == "OSOYOOS_INDIAN_RESERVE_NO_1" for h in hits)


def test_ftp_listing_dates_without_a_year_are_this_year_or_last():
    from datetime import date
    line = "-rw-rw-r--    1 ftp      ftp       2060332 Sep 21 20:09 X_SHP.zip"
    assert clss.listing_date(line, today=date(2026, 10, 26)) == "2026-09-21"
    assert clss.listing_date(line, today=date(2027, 2, 1)) == "2026-09-21"
    dated = "-rw-rw-r--    1 ftp      ftp       2060332 Sep 21  2024 X_SHP.zip"
    assert clss.listing_date(dated) == "2024-09-21"
