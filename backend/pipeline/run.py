"""Orchestrator: one GeoTIFF in, job folder out per the output contract.

CLI: python -m backend.pipeline.run <tile.tif> <output/job_dir>
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

from . import loader, terrain, render, hydrology, scoring

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


def run(tif_paths, out_dir, boundary=None):
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
    slope_deg, slope_pct, aspect_deg = terrain.slope_aspect(z, meta["res"])
    hs = terrain.hillshade(z, meta["res"])
    t = terrain.tpi(z, meta["res"])
    for a in (slope_deg, slope_pct, aspect_deg, hs, t):
        a[mask] = np.nan

    # phase 4: cold-air drainage (flow accumulation). Degrades to None if
    # WhiteboxTools is unavailable; the rest of the pipeline still runs.
    hydro_dir = os.path.join(out_dir, "_hydro")
    flowacc_log, flowacc_norm = hydrology.flow_accumulation(
        z, mask, meta, hydro_dir)

    # phase 5: frost risk + suitability composites (native grid).
    frost, suit, score_stats = scoring.frost_and_suitability(
        z, t, slope_deg, slope_pct, aspect_deg, flowacc_norm, mask, meta["res"])

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

    nodata_pct = 100.0 * float(mask.mean())
    stats = {
        "coverage": loader.coverage_stats(mask, meta["res"]),
        "elevation": terrain.elevation_stats(z, mask),
        "slope_pct": terrain.slope_stats(slope_pct, mask),
        "aspect_distribution_pct": terrain.aspect_distribution(
            aspect_deg, slope_deg, mask),
        "pooling": terrain.pooling_stats(t, mask),
        "plant_above_m": terrain.plant_above_line(z, t, mask),
        # phase 5 keys (frost, suitability_areas_ha/_acres)
        **score_stats,
        "flags": {
            "is_dsm": is_dsm,
            "canopy_caveat": canopy,
            "partial_dtm": (not is_dsm) and nodata_pct > 30.0,
            "has_drainage": flowacc_norm is not None,
        },
        "legends": legends,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # drop intermediates; the layer PNGs are the kept products
    shutil.rmtree(hydro_dir, ignore_errors=True)
    shutil.rmtree(mosaic_dir, ignore_errors=True)

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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m backend.pipeline.run <tile.tif> [more.tif ...] <out_dir>")
    *tifs, out = sys.argv[1:]
    s = run(tifs, out)
    print(json.dumps(s, indent=2))
