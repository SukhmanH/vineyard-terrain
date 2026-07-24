"""Terrain derivatives. All math on the NATIVE metric grid (UTM, 1 m)."""
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from matplotlib.colors import LightSource

POOLING_WINDOW_M = 200
POOLING_THRESH = -0.5
SINK_THRESH = -1.0


def _filled(z):
    zf = z.copy()
    zf[~np.isfinite(zf)] = np.nanmedian(z)
    return zf


def slope_aspect(z, res):
    zf = _filled(z)
    gy, gx = np.gradient(zf, res)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    slope_pct = 100.0 * np.tan(np.radians(slope_deg))
    aspect_deg = (np.degrees(np.arctan2(-gx, gy))) % 360.0
    return slope_deg, slope_pct, aspect_deg


def hillshade(z, res):
    zf = _filled(z)
    ls = LightSource(azdeg=315, altdeg=35)
    return ls.hillshade(zf, vert_exag=2, dx=res, dy=res)


def tpi(z, res, window_m=POOLING_WINDOW_M):
    zf = _filled(z)
    w = max(3, int(round(window_m / res)))
    t = zf - uniform_filter(zf, size=w, mode="nearest")
    return gaussian_filter(t, sigma=3)


def pooling_stats(t, mask):
    v = t[~mask]
    return {
        "window_m": POOLING_WINDOW_M,
        "pct_lt_-0.5": round(100.0 * float(np.mean(v < POOLING_THRESH)), 1),
        "pct_lt_-1.0": round(100.0 * float(np.mean(v < SINK_THRESH)), 1),
    }


def plant_above_line(z, t, mask, min_relief=80.0, band=10.0,
                     min_cells=500, safe_pct=10.0):
    """Lowest 10 m elevation band midpoint where pooling < safe_pct.
    Returns None on low-relief tiles where the concept is meaningless."""
    zv = z[~mask]
    tv = t[~mask]
    if zv.size == 0 or (np.nanmax(zv) - np.nanmin(zv)) < min_relief:
        return None
    lo = np.floor(np.nanmin(zv) / band) * band
    hi = np.ceil(np.nanmax(zv) / band) * band
    edges = np.arange(lo, hi + band, band)
    for i in range(len(edges) - 1):
        sel = (zv >= edges[i]) & (zv < edges[i + 1])
        if sel.sum() < min_cells:
            continue
        frac = 100.0 * float(np.mean(tv[sel] < POOLING_THRESH))
        if frac < safe_pct:
            return float((edges[i] + edges[i + 1]) / 2.0)
    return None


def aspect_distribution(aspect_deg, slope_deg, mask):
    """Pct of valid, non-flat ground facing each octant. Flat (<2 deg) has no
    meaningful aspect and is excluded."""
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    sel = (~mask) & (slope_deg >= 2.0)
    a = aspect_deg[sel]
    if a.size == 0:
        return {n: 0.0 for n in names}
    octant = np.floor(((a + 22.5) % 360.0) / 45.0).astype(int)
    return {n: round(100.0 * float(np.mean(octant == i)), 1)
            for i, n in enumerate(names)}


def elevation_stats(z, mask):
    v = z[~mask]
    return {
        "min": round(float(np.min(v)), 1),
        "median": round(float(np.median(v)), 1),
        "max": round(float(np.max(v)), 1),
        "relief_m": round(float(np.max(v) - np.min(v)), 1),
    }


def slope_stats(slope_pct, mask):
    v = slope_pct[~mask]
    return {"mean": round(float(np.mean(v)), 1),
            "p90": round(float(np.percentile(v, 90)), 1)}
