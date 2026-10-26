"""Orchestrator: one GeoTIFF in, job folder out per the output contract.

CLI: python -m backend.pipeline.run <tile.tif> <output/job_dir>

Two paths share the terrain math and the render:
  run()         manual uploads: a tile or a mosaic of tiles, optionally clipped to a
                boundary before any math (unchanged).
  run_parcel()  the parcel path: fetch the parcel's AOI (parcel + buffer), analyze it in
                context, and only then summarize inside the parcel. Clipping first deletes
                the upslope ground that drains cold air onto a parcel, turns an embankment
                just past the property line into an open edge, and makes the parcel's own
                lowest corner the valley floor.
"""
import functools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Polygon, box

from . import hydrology, loader, render, scoring, terrain, tiles, zonal
from . import parcel as parcels

LAYER_SPECS = [
    # (name, cmap, kind)  kind: "seq" sequential, "div" diverging, "fixed"
    ("elevation", "terrain", "seq"),
    ("hillshade", "gray", "fixed01"),
    ("slope", "magma", "slope"),
    ("aspect", "hsv", "aspect"),
    ("tpi", "RdBu", "div"),
    ("flowacc", "Blues", "seq"),
    ("frost", "RdYlGn_r", "pct"),
    ("suitability", "RdYlGn", "pct"),
]

# AOI ring around a parcel (VTA_PARCEL_BUFFER_M overrides). 750 m moved Hwy 97's high-risk
# share 3.7 points against 1,000 m; from 1,000 m out the six properties move <= 0.7.
DEFAULT_BUFFER_M = 1000.0
OFF_AXIS_BUFFER_M = 1250.0   # off the valley-axis table the AOI must reach the real floor
WIDEN_FACTOR = 2.0           # one widening when cold air enters across the AOI edge
MAX_BUFFER_M = 2000.0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParcelAnalysisError(RuntimeError):
    """The parcel path could not produce scores comparable with other listings."""


def run(tif_paths, out_dir, boundary=None):
    """Analyze a tile or mosaic, always clearing intermediates afterwards.

    The hydrology and mosaic scratch dirs are large: a failed five-tile job
    left 2 GB of _hydro behind. Cleanup runs in a finally so a crash cannot
    fill the disk on a long-lived server.
    """
    try:
        return _run(tif_paths, out_dir, boundary=boundary)
    finally:
        shutil.rmtree(os.path.join(out_dir, "_hydro"), ignore_errors=True)
        shutil.rmtree(os.path.join(out_dir, "_mosaic"), ignore_errors=True)


def _terrain_layers(z, mask, res):
    """Slope, aspect, hillshade and TPI on the native grid, NaN at NoData."""
    slope_deg, slope_pct, aspect_deg = terrain.slope_aspect(z, res)
    hs = terrain.hillshade(z, res)
    t = terrain.tpi(z, res)
    for a in (slope_deg, slope_pct, aspect_deg, hs, t):
        a[mask] = np.nan
    return slope_deg, slope_pct, aspect_deg, hs, t


def _terrain_stats(z, mask, meta, slope_deg, slope_pct, aspect_deg, t):
    return {
        "coverage": loader.coverage_stats(mask, meta["res"]),
        "elevation": terrain.elevation_stats(z, mask),
        "slope_pct": terrain.slope_stats(slope_pct, mask),
        "aspect_distribution_pct": terrain.aspect_distribution(
            aspect_deg, slope_deg, mask),
        "pooling": terrain.pooling_stats(t, mask),
        "plant_above_m": terrain.plant_above_line(z, t, mask),
    }


def _render_layers(arrays, meta, out_dir):
    """Warp each layer to EPSG:3857, write its PNG and bounds.json. Layers whose source
    array is absent are skipped. Returns (legends, wgs84 bounds)."""
    os.makedirs(os.path.join(out_dir, "layers"), exist_ok=True)
    legends = {}
    web_bounds = None
    for name, cmap, kind in LAYER_SPECS:
        arr = arrays.get(name)
        if arr is None:
            continue
        warped, wb = render.warp_to_web(arr, meta)
        web_bounds = wb
        png = os.path.join(out_dir, "layers", f"{name}.png")
        if kind == "div":
            legends[name] = render.save_diverging_png(warped, png, cmap)
        elif kind == "fixed01":
            legends[name] = render.save_layer_png(warped, png, cmap, 0.0, 1.0)
        elif kind == "slope":
            legends[name] = render.save_layer_png(warped, png, cmap, 0.0, 45.0)
        elif kind == "aspect":
            legends[name] = render.save_layer_png(warped, png, cmap, 0.0, 360.0)
        elif kind == "pct":
            legends[name] = render.save_layer_png(warped, png, cmap, 0.0, 100.0)
        else:
            legends[name] = render.save_layer_png(warped, png, cmap)

    bounds = render.web_bounds_to_wgs(web_bounds)
    with open(os.path.join(out_dir, "bounds.json"), "w") as f:
        json.dump(bounds, f, indent=2)
    return legends, bounds


def _run(tif_paths, out_dir, boundary=None):
    """Analyze one tile or a mosaic of tiles into a job folder.

    tif_paths: a single path (str) or a list of paths. Multiple tiles are
    reprojected to the first tile's CRS and merged before analysis.
    boundary: optional GeoJSON polygon (dict) clipping the analysis to a parcel.
    """
    if isinstance(tif_paths, str):
        tif_paths = [tif_paths]
    os.makedirs(os.path.join(out_dir, "layers"), exist_ok=True)

    mosaic_dir = os.path.join(out_dir, "_mosaic")
    z, mask, meta = loader.load_mosaic(tif_paths, workdir=mosaic_dir)

    # a mosaic is a DSM (canopy caveat) if ANY input tile is a surface model
    flags = [loader.detect_model_type(p) for p in tif_paths]
    is_dsm = any(f[0] for f in flags)
    canopy = any(f[1] for f in flags)

    # optional parcel-boundary clip (phase 6)
    if boundary is not None:
        z, mask = loader.clip_to_boundary(z, mask, meta, boundary)

    # terrain math on native grid
    slope_deg, slope_pct, aspect_deg, hs, t = _terrain_layers(z, mask, meta["res"])

    # field-scale surface shared by routing and scoring, and hydro-flattened
    # water, which is neither routed over nor scored
    field = scoring.field_surface(z, mask, meta["res"])
    water = scoring.water_mask(z, mask, meta["res"])

    # phase 4: cold-air routing (contributing area, closed-hollow depth).
    # Degrades to None if WhiteboxTools is unavailable; the rest still runs.
    hydro_dir = os.path.join(out_dir, "_hydro")
    routing = hydrology.cold_air_routing(
        field["z"], mask | water, meta, hydro_dir)
    flowacc_log = routing["flowacc_log"] if routing is not None else None

    # phase 5: frost risk + suitability composites (native grid).
    geo = scoring.block_geography(meta)
    frost, suit, score_stats = scoring.frost_and_suitability(
        z, mask, meta["res"], lat=geo["lat"], field=field, routing=routing,
        water=water, floor_rows=geo["floor_rows"])
    has_drainage = routing is not None
    del field, water, routing   # mosaics are large; render needs the memory

    # display artifacts. Layers whose source array is absent are skipped.
    arrays = {
        "elevation": z,
        "hillshade": hs,
        "slope": slope_deg,
        "aspect": aspect_deg,
        "tpi": t,
        "flowacc": flowacc_log,
        "frost": frost,
        "suitability": suit,
    }
    legends, _ = _render_layers(arrays, meta, out_dir)

    nodata_pct = 100.0 * float(mask.mean())
    stats = {
        **_terrain_stats(z, mask, meta, slope_deg, slope_pct, aspect_deg, t),
        # phase 5 keys (frost, suitability_areas_ha/_acres)
        **score_stats,
        "flags": {
            "is_dsm": is_dsm,
            "canopy_caveat": canopy,
            "partial_dtm": (not is_dsm) and nodata_pct > 30.0,
            "has_drainage": has_drainage,
        },
        "legends": legends,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    names = [os.path.basename(p) for p in tif_paths]
    meta_out = {
        "job_id": os.path.basename(out_dir.rstrip("/")),
        "filename": names[0] if len(names) == 1 else names,
        "tile_count": len(names),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "is_dsm": is_dsm,
        "canopy_caveat": canopy,
        "clipped": boundary is not None,
        "crs": str(meta["crs"]),
        "res_m": meta["res"],
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    return stats


# ------------------------------------------------------------------ parcel path
@functools.lru_cache(maxsize=1)
def git_sha():
    """Commit the API runs (VTA_GIT_SHA in a container, else git), for provenance."""
    if os.environ.get("VTA_GIT_SHA"):
        return os.environ["VTA_GIT_SHA"]
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if sha and dirty else (sha or "unknown")


def choose_buffer(geom_3005, requested=None):
    """(buffer_m, off_axis). Off the valley-axis table the block finds its own floor, so
    the AOI must be wide enough to reach the real valley floor below the parcel."""
    base = float(requested or os.environ.get("VTA_PARCEL_BUFFER_M") or DEFAULT_BUFFER_M)
    c = geom_3005.representative_point()
    lons, lats = transform("EPSG:3005", "EPSG:4326", [c.x], [c.y])
    floor, _ = scoring.valley_floor_profile(np.asarray(lats), np.asarray(lons))
    off_axis = not np.isfinite(floor[0])
    return (max(base, OFF_AXIS_BUFFER_M) if off_axis else base), off_axis


def analyze_in_context(z, mask, meta, parcel_3005, workdir):
    """The full analysis over the AOI, then the parcel located on that grid.

    Routing is required here, not optional as in the manual path: without the two routing
    mechanisms a parcel's score would not be comparable with other listings.
    """
    res = meta["res"]
    slope_deg, slope_pct, aspect_deg, hs, t = _terrain_layers(z, mask, res)
    tstats = _terrain_stats(z, mask, meta, slope_deg, slope_pct, aspect_deg, t)
    field = scoring.field_surface(z, mask, res)
    water = scoring.water_mask(z, mask, res)
    routing = hydrology.cold_air_routing(field["z"], mask | water, meta, workdir)
    if routing is None:
        raise ParcelAnalysisError(
            "cold-air routing (WhiteboxTools) failed; without it this parcel's score "
            "would not be comparable with other listings")
    geo = scoring.block_geography(meta)
    detail = scoring.score_detail(z, mask, res, lat=geo["lat"], field=field,
                                  routing=routing, water=water, floor_rows=geo["floor_rows"])
    if detail is None:
        raise ParcelAnalysisError("no bare-earth land in the analysed area")
    inside = tiles.parcel_mask(parcel_3005, meta)
    if not inside.any():
        raise ParcelAnalysisError("the parcel does not overlap the analysed grid")

    # does cold air from beyond the AOI (or from a NoData gap) drain onto the parcel?
    rim, hole = zonal.inflow_cells(field["z"], ~mask, water, res)
    target = inside & detail["land"]
    edge_pct = nodata_pct = 0.0
    if rim.any():
        edge_pct = zonal.reached_pct(hydrology.edge_inflow(
            routing["breached_path"], rim, mask | water, meta, workdir), target)
    if hole.any():
        nodata_pct = zonal.reached_pct(hydrology.edge_inflow(
            routing["breached_path"], hole, mask | water, meta, workdir), target)

    return {
        "terrain_stats": tstats,
        "block_stats": scoring.block_stats(detail, res),
        "detail": detail,
        "field_z": field["z"],
        "inside": inside,
        "edge_inflow_pct": edge_pct,
        "nodata_inflow_pct": nodata_pct,
        "arrays": {
            "elevation": z, "hillshade": hs, "slope": slope_deg, "aspect": aspect_deg,
            "tpi": t, "flowacc": routing["flowacc_log"], "frost": detail["frost"],
            "suitability": detail["suit"],
        },
    }


def _subgrid(meta, r0, r1, c0, c1):
    t = meta["transform"] * Affine.translation(c0, r0)
    res = meta["res"]
    return {"crs": meta["crs"], "res": res, "width": c1 - c0, "height": r1 - r0,
            "transform": t,
            "bounds": BoundingBox(t.c, t.f - res * (r1 - r0), t.c + res * (c1 - c0), t.f)}


def _canopy(entries, info, z, meta, inside):
    """DSM minus DEM over the parcel where every DEM tile under it has a DSM twin of the
    same delivery. Informational only (orchard trees come out before planting), so any
    failure here leaves canopy unknown rather than failing the job."""
    rows, cols = np.nonzero(inside)
    r0, r1 = max(int(rows.min()) - 5, 0), min(int(rows.max()) + 6, meta["height"])
    c0, c1 = max(int(cols.min()) - 5, 0), min(int(cols.max()) + 6, meta["width"])
    sub = _subgrid(meta, r0, r1, c0, c1)
    window = box(*transform_bounds(meta["crs"], "EPSG:3005", *sub["bounds"]))
    by_key = {e["key"]: e for e in entries}
    twins = []
    for key in info["tile_keys"]:
        dem = by_key.get(key)
        if dem is None or not Polygon(dem["footprint_3005"]).intersects(window):
            continue
        twin = tiles.dsm_twin(entries, dem)
        if twin is None:
            return None, []          # a partial canopy would undercount
        twins.append(twin)
    if not twins:
        return None, []
    try:
        paths = [tiles.fetch(t, window.bounds) for t in twins]
        dsm = tiles.assemble(paths, sub)
    except (tiles.TileError, OSError, ValueError):
        return None, []
    canopy = np.full(z.shape, np.nan, dtype="float32")
    canopy[r0:r1, c0:c1] = dsm - z[r0:r1, c0:c1]
    return canopy, [t["key"] for t in twins]


def run_parcel(resolution, out_dir, buffer_m=None, entries=None, on_status=None,
               label=None):
    """The parcel path. resolution is a resolved parcel.resolve() result.

    AOI = parcel + buffer, analyzed in context and summarized inside the parcel. Cold air
    entering across the AOI's upslope edge onto the parcel widens the AOI once; if it
    still does, the result is flagged buffer_inadequate. Returns stats (with its `parcel`
    block) and writes the output contract plus parcel.geojson.
    """
    if resolution.get("status") != "resolved":
        raise ParcelAnalysisError("the parcel is not resolved")
    status = on_status or (lambda s: None)
    geom = resolution["geometry_3005"]
    buffer, off_axis = choose_buffer(geom, buffer_m)
    os.makedirs(out_dir, exist_ok=True)
    workdir = os.path.join(out_dir, "_hydro")
    widened = False
    try:
        while True:
            aoi = geom.buffer(buffer)
            # checked before anything is downloaded, the tile index included
            if not parcels.inside_service_area(aoi):
                raise ParcelAnalysisError("the AOI reaches outside the southern interior "
                                          "area this service covers")
            if entries is None:
                entries = tiles.load_index()
            status("fetching")
            z, mask, meta, info = tiles.load_aoi(aoi, geom, entries)
            status("analyzing")
            shutil.rmtree(workdir, ignore_errors=True)
            a = analyze_in_context(z, mask, meta, geom, workdir)
            if ((a["edge_inflow_pct"] or 0.0) > zonal.EDGE_INFLOW_MAX_PCT
                    and not widened and buffer < MAX_BUFFER_M):
                buffer, widened = min(buffer * WIDEN_FACTOR, MAX_BUFFER_M), True
                del a, z, mask
                continue
            break

        canopy, dsm_keys = _canopy(entries, info, z, meta, a["inside"])
        metrics = zonal.parcel_metrics(
            a["detail"], a["field_z"], a["inside"], ~mask, meta["res"],
            parcel_confidence=resolution["confidence"], parcel_flags=resolution["flags"],
            plant_above_m=a["terrain_stats"]["plant_above_m"], canopy=canopy,
            edge_inflow_pct=a["edge_inflow_pct"], nodata_inflow_pct=a["nodata_inflow_pct"],
            widened=widened, mix=info["mix"], aoi_resampled_m=info["aoi_resampled_m"])
        metrics = {
            "pid": resolution["pid"], "pids": resolution["pids"],
            "area_acres": resolution["area_acres"], "parcel_source": resolution["source"],
            "parcel_confidence": resolution["confidence"], **metrics,
        }
        aoi_wgs = transform_bounds("EPSG:3005", "EPSG:4326", *aoi.bounds)
        provenance = {
            "tile_keys": info["tile_keys"], "years": info["years"], "product": "dem",
            "crs": info["crs"], "res_m": info["res_m"],
            "aoi_resampled_m": info["aoi_resampled_m"], "buffer_m": buffer,
            "buffer_widened": widened, "off_axis": off_axis,
            "aoi_bounds_wgs84": [round(v, 6) for v in aoi_wgs],
            "aoi_cells": int(meta["width"] * meta["height"]),
            "parcel_valid_pct": info["parcel_valid_pct"],
            "aoi_valid_pct": info["aoi_valid_pct"],
            "deliveries_considered": info["deliveries_considered"], "mix": info["mix"],
            "dsm_keys": dsm_keys, "model_version": zonal.MODEL_VERSION,
            "frost_model": scoring.MODEL, "git_sha": git_sha(),
        }
        arrays, block, tstats = a["arrays"], a["block_stats"], a["terrain_stats"]
        del a, canopy                 # free the scoring grids before the render
        legends, _ = _render_layers(arrays, meta, out_dir)
        del arrays

        nodata_pct = 100.0 * float(mask.mean())
        stats = {
            **tstats, **block,
            "flags": {"is_dsm": False, "canopy_caveat": False,
                      "partial_dtm": nodata_pct > 30.0, "has_drainage": True},
            "legends": legends,
            "parcel": metrics,
        }
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        meta_out = {
            "job_id": os.path.basename(out_dir.rstrip("/\\")),
            "filename": label or resolution["pid"] or "parcel",
            "tile_count": len(info["tile_keys"]),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "is_dsm": False, "canopy_caveat": False, "clipped": False,
            "crs": str(meta["crs"]), "res_m": meta["res"],
            "kind": "parcel", "product": "dem", "provenance": provenance,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2)
        with open(os.path.join(out_dir, "parcel.geojson"), "w") as f:
            json.dump(parcels.feature_collection(resolution, aoi), f)
        tiles.evict(keep=info["local_paths"])
        return stats, provenance
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m backend.pipeline.run <tile.tif> [more.tif ...] <out_dir>")
    *tifs, out = sys.argv[1:]
    s = run(tifs, out)
    print(json.dumps(s, indent=2))
