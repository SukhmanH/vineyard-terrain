"""Warp native-grid arrays to EPSG:3857 and write colormapped RGBA PNGs.

Math happens on the native UTM grid (terrain.py). This module only prepares
display artifacts for Leaflet: one PNG per layer plus WGS84 corner bounds.
"""
import numpy as np
import rasterio
from rasterio.warp import (calculate_default_transform, reproject,
                           Resampling, transform_bounds)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

WEB = "EPSG:3857"
WGS = "EPSG:4326"


def warp_to_web(arr, meta):
    """Reproject a float array from the tile CRS to EPSG:3857.
    Returns (warped_array_with_nan, web_bounds)."""
    src_crs = meta["crs"]
    b = meta["bounds"]
    transform, width, height = calculate_default_transform(
        src_crs, WEB, meta["width"], meta["height"],
        b.left, b.bottom, b.right, b.top)
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=arr.astype("float32"),
        destination=dst,
        src_transform=meta["transform"],
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=WEB,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return dst, (left, bottom, right, top)


def web_bounds_to_wgs(web_bounds):
    w, s, e, n = transform_bounds(WEB, WGS, *web_bounds)
    return {"west": w, "south": s, "east": e, "north": n}


def save_layer_png(arr_web, out_png, cmap, vmin=None, vmax=None):
    """Colormap a warped array to RGBA uint8 with alpha 0 at NaN."""
    a = arr_web
    finite = np.isfinite(a)
    if vmin is None:
        vmin = float(np.nanpercentile(a, 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(a, 98)) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = matplotlib.colormaps[cmap](norm(np.where(finite, a, vmin)))
    rgba[..., 3] = np.where(finite, 1.0, 0.0)
    plt.imsave(out_png, (rgba * 255).astype("uint8"))
    return {"vmin": round(vmin, 2), "vmax": round(vmax, 2), "cmap": cmap}


def save_diverging_png(arr_web, out_png, cmap="RdBu", pct=98):
    """Symmetric colormap around 0 (for TPI)."""
    finite = np.isfinite(arr_web)
    vlim = float(np.nanpercentile(np.abs(arr_web), pct)) if finite.any() else 1.0
    return save_layer_png(arr_web, out_png, cmap, vmin=-vlim, vmax=vlim)
