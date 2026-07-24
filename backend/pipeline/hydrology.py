"""Phase 4. Cold-air drainage via WhiteboxTools.

On clear calm nights cold air flows downslope like water and converges in the
same channels and hollows that surface water would. We reuse hydrologic flow
routing to find those convergence zones: breach depressions so drainage is not
blocked by 1 m LiDAR speckle, then run D-infinity flow accumulation. High
accumulation marks where cold air (and frost risk) concentrates.

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


def flow_accumulation(z, mask, meta, workdir):
    """Run breach + D-inf flow accumulation on the native grid.

    Returns (flowacc_log, flowacc_norm):
      flowacc_log  - log1p(flow accumulation), NaN at NoData, for display.
      flowacc_norm - flowacc_log scaled by its 95th percentile and clipped to
                     [0, 1] over valid cells, for scoring.
    On any Whitebox failure returns (None, None) so the pipeline can continue
    without the drainage component rather than aborting the whole job.
    """
    try:
        import whitebox
    except ImportError:
        return None, None

    os.makedirs(workdir, exist_ok=True)
    dem = os.path.abspath(os.path.join(workdir, "dem.tif"))
    breached = os.path.abspath(os.path.join(workdir, "breached.tif"))
    flowacc = os.path.abspath(os.path.join(workdir, "flowacc.tif"))

    _write_dem(z, mask, meta, dem)

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    try:
        # breach (carve) rather than fill: preferred for high-res LiDAR, keeps
        # channels continuous without raising whole basins.
        rc1 = wbt.breach_depressions_least_cost(dem, breached, dist=100)
        rc2 = wbt.d_inf_flow_accumulation(breached, flowacc, out_type="cells")
    except Exception:
        return None, None
    if rc1 != 0 or rc2 != 0 or not os.path.exists(flowacc):
        return None, None

    with rasterio.open(flowacc) as ds:
        acc = ds.read(1).astype("float32")
        nd = ds.nodata

    valid = ~mask
    if nd is not None:
        valid &= (acc != nd)
    valid &= np.isfinite(acc)

    flowacc_log = np.full(acc.shape, np.nan, dtype="float32")
    flowacc_log[valid] = np.log1p(acc[valid])

    if not valid.any():
        return flowacc_log, np.zeros(acc.shape, dtype="float32")

    p95 = float(np.nanpercentile(flowacc_log[valid], 95))
    if p95 <= 0:
        p95 = float(np.nanmax(flowacc_log[valid])) or 1.0
    norm = np.clip(np.nan_to_num(flowacc_log, nan=0.0) / p95, 0.0, 1.0)
    norm[~valid] = 0.0
    return flowacc_log, norm.astype("float32")
