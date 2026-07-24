"""Load a LidarBC GeoTIFF DEM tile into array + mask + meta."""
import os
import tempfile

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Canonical NoData for our staged/merged rasters. LidarBC uses -32767.0.
_ND = -32767.0


def _meta_from_dataset(ds):
    return {
        "crs": ds.crs,
        "transform": ds.transform,
        "res": float(ds.res[0]),
        "bounds": ds.bounds,
        "nodata": ds.nodata,
        "width": ds.width,
        "height": ds.height,
    }


def _array_and_mask(z, nodata):
    mask = ~np.isfinite(z) if nodata is None else ((z == nodata) | ~np.isfinite(z))
    z = z.copy()
    z[mask] = np.nan
    return z, mask


def load_dem(path):
    """Return (z, mask, meta). z has NaN at NoData; mask True where invalid."""
    with rasterio.open(path) as ds:
        z = ds.read(1).astype("float32")
        meta = _meta_from_dataset(ds)
    z, mask = _array_and_mask(z, meta["nodata"])
    return z, mask, meta


def _reproject_to(path, dst_crs, workdir, idx):
    """Reproject one tile to dst_crs, writing a temp GeoTIFF. Returns its path.
    If already in dst_crs, returns the original path unchanged."""
    with rasterio.open(path) as ds:
        if ds.crs == dst_crs:
            return path
        transform, width, height = calculate_default_transform(
            ds.crs, dst_crs, ds.width, ds.height, *ds.bounds)
        nd = ds.nodata if ds.nodata is not None else _ND
        profile = ds.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width,
                       height=height, nodata=nd, driver="GTiff", dtype="float32")
        out = os.path.join(workdir, f"reproj_{idx}.tif")
        with rasterio.open(out, "w", **profile) as dst:
            reproject(
                source=rasterio.band(ds, 1),
                destination=rasterio.band(dst, 1),
                src_transform=ds.transform, src_crs=ds.crs,
                dst_transform=transform, dst_crs=dst_crs,
                src_nodata=ds.nodata, dst_nodata=nd,
                resampling=Resampling.bilinear,
            )
    return out


def load_mosaic(paths, workdir=None):
    """Merge several DEM tiles into one array + mask + meta.

    Tiles may have different CRS (real LidarBC neighbours mix EPSG:6654 and
    2955). We reproject every tile to the FIRST tile's CRS before merging,
    since terrain math must run on a single consistent metric grid.

    Returns (z, mask, meta) matching load_dem. A single-path list short
    circuits to load_dem so the common case stays simple.
    """
    if len(paths) == 1:
        return load_dem(paths[0])

    cleanup = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="mosaic_")
    os.makedirs(workdir, exist_ok=True)
    try:
        with rasterio.open(paths[0]) as ds0:
            dst_crs = ds0.crs
        reprojected = [_reproject_to(p, dst_crs, workdir, i)
                       for i, p in enumerate(paths)]

        datasets = [rasterio.open(p) for p in reprojected]
        try:
            arr, out_transform = merge(datasets, nodata=_ND)
            z = arr[0].astype("float32")
            res = float(datasets[0].res[0])
            height, width = z.shape
            # bounds from the merged transform
            left, top = out_transform.c, out_transform.f
            right = left + out_transform.a * width
            bottom = top + out_transform.e * height
            bounds = rasterio.coords.BoundingBox(left, bottom, right, top)
            meta = {
                "crs": dst_crs, "transform": out_transform, "res": res,
                "bounds": bounds, "nodata": _ND, "width": width,
                "height": height,
            }
        finally:
            for d in datasets:
                d.close()
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    z, mask = _array_and_mask(z, _ND)
    return z, mask, meta


def clip_to_boundary(z, mask, meta, geojson):
    """Mask out cells outside a GeoJSON parcel polygon.

    geojson may be a Feature, FeatureCollection, or bare geometry in the
    tile's own CRS or WGS84 (we reproject the geometry to the tile CRS).
    Returns a new (z, mask) with everything outside the polygon set NoData.
    Cells already NoData stay NoData. On any failure returns the inputs
    unchanged so a bad boundary never sinks the job.
    """
    try:
        from rasterio.features import geometry_mask
        from rasterio.warp import transform_geom
    except ImportError:
        return z, mask

    geoms = _extract_geometries(geojson)
    if not geoms:
        return z, mask

    # boundaries are typically drawn in WGS84 (Leaflet); reproject to tile CRS.
    try:
        geoms = [transform_geom("EPSG:4326", meta["crs"], g) for g in geoms]
    except Exception:
        pass  # assume already in tile CRS

    try:
        outside = geometry_mask(
            geoms, out_shape=(meta["height"], meta["width"]),
            transform=meta["transform"], invert=False)
    except Exception:
        return z, mask

    new_mask = mask | outside
    z = z.copy()
    z[new_mask] = np.nan
    return z, new_mask


def _extract_geometries(geojson):
    """Pull geometry dicts out of a Feature/FeatureCollection/geometry."""
    if not isinstance(geojson, dict):
        return []
    t = geojson.get("type")
    if t == "FeatureCollection":
        return [f["geometry"] for f in geojson.get("features", [])
                if f.get("geometry")]
    if t == "Feature":
        return [geojson["geometry"]] if geojson.get("geometry") else []
    if t in ("Polygon", "MultiPolygon"):
        return [geojson]
    return []


def detect_model_type(filename):
    """Return (is_dsm, canopy_caveat) from LidarBC naming conventions."""
    name = os.path.basename(filename).lower()
    if "dtm" in name:
        return False, False
    if "dsm" in name:
        return True, True
    # older flights lack the suffix; assume possible surface model
    return True, True


def coverage_stats(mask, res):
    valid = int((~mask).sum())
    ha = valid * res * res / 10000.0
    return {
        "valid_ha": round(ha, 1),
        "valid_acres": round(ha * 2.471, 1),
        "nodata_pct": round(100.0 * float(mask.mean()), 1),
    }
