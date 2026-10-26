"""Phase 4. Cold-air routing via WhiteboxTools.

On clear calm nights cold air flows downslope like water and converges in the
same channels and hollows that surface water would. We reuse hydrologic tools
to find those places:
  - fill depressions: fill minus surface is how deep cold air ponds in a
    closed hollow before it spills over the rim (frost pockets, and slopes
    dammed by road embankments or berms).
  - breach depressions so routing is not blocked by LiDAR speckle, then run
    D-infinity to get specific contributing area: how much upslope ground
    drains through each cell.

Routing runs on the field-scale surface (scoring.field_surface), not the raw
1 m DEM: a cold-air layer metres thick follows the terrain at that scale, not
decimetre ruts and vine rows. Hydro-flattened water is staged as NoData so
lakes act as outlets, which is also how cold air behaves over open water, and
so routing never has to resolve a 50 ha exact flat.

WhiteboxTools is file based and downloads its binary on first run (needs the
network once). It only reads and writes real files, so we stage a DEM built
from the loaded array (never the raw upload, which may be a DSM) into a temp
dir and hand Whitebox ABSOLUTE paths.
"""
import os

import numpy as np
import rasterio

# Sentinel NoData for the staged DEM. Whitebox routes over real nodata, but a
# NaN inside a GeoTIFF confuses its flow tools, so we use a numeric sentinel.
_ND = -32767.0


def _write_dem(z, mask, meta, path):
    """Write the native-grid elevation array as a GeoTIFF for Whitebox."""
    zf = np.where(mask, _ND, z).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=meta["height"], width=meta["width"], count=1,
        dtype="float32", crs=meta["crs"], transform=meta["transform"],
        nodata=_ND,
    ) as ds:
        ds.write(zf, 1)


def _read(path, valid):
    """Read a Whitebox output as float32 with NaN wherever it has no data."""
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float32")
        nd = ds.nodata
    ok = valid & np.isfinite(a)
    if nd is not None:
        ok &= a != nd
    a[~ok] = np.nan
    return a


def cold_air_routing(z, mask, meta, workdir):
    """Fill + breach + D-inf on the native grid.

    z: field-scale surface; mask: True where there is no ground to route over
    (NoData and water). Returns a dict of float32 grids, NaN where masked:
      sca         D-inf specific contributing area (m).
      sink_depth  depth below the spill point of a closed depression (m),
                  0 outside depressions.
      flowacc_log log1p(sca), for the display layer.
    On any Whitebox failure returns None so the pipeline can continue without
    the routing mechanisms rather than aborting the whole job.
    """
    try:
        import whitebox
    except ImportError:
        return None

    os.makedirs(workdir, exist_ok=True)
    dem = os.path.abspath(os.path.join(workdir, "dem.tif"))
    filled = os.path.abspath(os.path.join(workdir, "filled.tif"))
    breached = os.path.abspath(os.path.join(workdir, "breached.tif"))
    sca_path = os.path.abspath(os.path.join(workdir, "sca.tif"))

    _write_dem(z, mask, meta, dem)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    try:
        # fill without flat fixing so a filled hollow is exactly level and
        # fill minus surface is the true ponding depth
        rc1 = wbt.fill_depressions(dem, filled, fix_flats=False)
        # breach (carve) rather than fill for routing: preferred for
        # high-res LiDAR, keeps channels continuous without raising basins.
        rc2 = wbt.breach_depressions_least_cost(dem, breached, dist=100)
        rc3 = wbt.d_inf_flow_accumulation(breached, sca_path, out_type="sca")
    except Exception:
        return None
    if rc1 != 0 or rc2 != 0 or rc3 != 0 or not (
            os.path.exists(filled) and os.path.exists(sca_path)):
        return None

    valid = ~mask
    sca = _read(sca_path, valid)
    depth = _read(filled, valid) - np.where(valid, z, np.nan)
    depth = np.where(np.isfinite(depth), np.clip(depth, 0.0, None), np.nan)
    flowacc_log = np.where(np.isfinite(sca), np.log1p(np.clip(sca, 0.0, None)),
                           np.nan)
    # the breached surface stays in workdir (the caller removes it) for edge_inflow
    return {"sca": sca, "sink_depth": depth, "flowacc_log": flowacc_log,
            "breached_path": breached}


def edge_inflow(breached_path, loading, mask, meta, workdir):
    """Where cold air entering across given cells ends up: D-inf mass flux of a loading
    grid (1 on the cells it enters across, 0 elsewhere) over the breached routing surface.

    Returns float32 flux (in cells of loading draining through each cell, NaN where
    masked), or None if WhiteboxTools fails. The parcel path uses it to tell whether the
    AOI cut off upslope ground that drains onto the parcel.
    """
    try:
        import whitebox
    except ImportError:
        return None
    load_p = os.path.abspath(os.path.join(workdir, "edge_loading.tif"))
    eff_p = os.path.abspath(os.path.join(workdir, "edge_efficiency.tif"))
    abs_p = os.path.abspath(os.path.join(workdir, "edge_absorption.tif"))
    out_p = os.path.abspath(os.path.join(workdir, "edge_flux.tif"))
    ones = np.ones(loading.shape, dtype="float32")
    _write_dem(loading.astype("float32"), mask, meta, load_p)
    _write_dem(ones, mask, meta, eff_p)
    _write_dem(0.0 * ones, mask, meta, abs_p)
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    try:
        rc = wbt.d_inf_mass_flux(breached_path, load_p, eff_p, abs_p, out_p)
    except Exception:
        return None
    if rc != 0 or not os.path.exists(out_p):
        return None
    return _read(out_p, ~mask)
