"""Synthetic LidarBC-shaped tiles for the parcel-path tests.

Real NAD83(CSRS) UTM 11N coordinates near Oliver (or west of the valley for off-axis cases),
so footprints, CRS handling, the valley-axis table and the AOI grid behave as they do on
real tiles.
"""
import os

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform
from shapely.geometry import Polygon

from backend.pipeline import tiles

ND = -32767.0
X0, Y0 = 312000.0, 5452000.0        # Oliver, EPSG:2955: on the valley-axis table
X_OFF_AXIS = 270000.0               # ~ -120.15: west of the table, inside the service area


def utm_box(x0, y0, x1, y1, crs="EPSG:2955"):
    """A UTM rectangle as a shapely polygon in EPSG:3005."""
    fx, fy = transform(crs, "EPSG:3005", [x0, x1, x1, x0], [y1, y1, y0, y0])
    return Polygon(zip(fx, fy))


def cells_box(x0, y_top, r0, r1, c0, c1):
    """Rows r0:r1, cols c0:c1 of a 1 m grid with origin (x0, y_top), in EPSG:3005."""
    return utm_box(x0 + c0, y_top - r1, x0 + c1, y_top - r0)


def grid_meta(x0, y_top, ny, nx, crs="EPSG:2955"):
    return {"crs": CRS.from_user_input(crs), "transform": from_origin(x0, y_top, 1, 1),
            "res": 1.0, "width": nx, "height": ny,
            "bounds": BoundingBox(x0, y_top - ny, x0 + nx, y_top), "nodata": ND}


def write_array(path, z, x0, y_top, crs="EPSG:2955"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ny, nx = z.shape
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1,
                       dtype="float32", crs=crs, transform=from_origin(x0, y_top, 1, 1),
                       nodata=ND) as ds:
        ds.write(np.where(np.isfinite(z), z, ND).astype("float32"), 1)
    return path


def write_tile(path, x0, y_top, nx, ny, zfun, crs="EPSG:2955", nodata_where=None):
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float64")
    X, Y = x0 + xx + 0.5, y_top - yy - 0.5
    z = zfun(X, Y).astype("float32")
    if nodata_where is not None:
        z[nodata_where(X, Y)] = ND
    return write_array(path, z, x0, y_top, crs)


def entry_for(path, year, name, product="dem", lm="2026-01-01T00:00:00Z"):
    """An index entry for a local tile, as tiles.refresh_index would write it."""
    e = tiles.parse_key(f"082/082e/{year}/{product}/{name}")
    with rasterio.open(path) as ds:
        e.update(tiles.header_of(ds))
    e.update(size=os.path.getsize(path), etag="0" * 32, last_modified=lm)
    return e


def rough(z, seed=0):
    """Millimetre noise: real LiDAR ground is never exactly level, and an exactly level
    synthetic plane would be classed as hydro-flattened water."""
    rng = np.random.default_rng(seed)
    return (z + rng.normal(0.0, 0.002, z.shape)).astype("float32")
