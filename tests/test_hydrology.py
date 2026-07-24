"""Phase-4 hydrology tests. These exercise the real WhiteboxTools binary
(downloaded on first use). If Whitebox cannot run, the module returns
(None, None) and the tests skip rather than fail, since the pipeline is
designed to degrade without it."""
import numpy as np
import pytest

from backend.pipeline import hydrology


def _channel_dem(ny=120, nx=120):
    """A plane draining north with a single N-S convergence channel."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    z = 300.0 + 0.05 * yy
    z -= 6.0 * np.exp(-(((xx - nx / 2) ** 2) / (2 * 4.0 ** 2)))
    mask = np.zeros((ny, nx), dtype=bool)
    from rasterio.transform import from_origin
    meta = {
        "crs": "EPSG:2955",
        "transform": from_origin(300000.0, 5500000.0, 1.0, 1.0),
        "res": 1.0, "width": nx, "height": ny, "nodata": -32767.0,
    }
    return z.astype("float32"), mask, meta


def test_flow_accumulation_finds_channel(tmp_path):
    z, mask, meta = _channel_dem()
    flowacc_log, flowacc_norm = hydrology.flow_accumulation(
        z, mask, meta, str(tmp_path / "hydro"))
    if flowacc_log is None:
        pytest.skip("WhiteboxTools unavailable in this environment")

    # contract: log array is NaN only where masked; norm is 0..1
    assert flowacc_log.shape == z.shape
    assert np.isfinite(flowacc_log[~mask]).all()
    assert np.nanmin(flowacc_norm) >= 0.0 and np.nanmax(flowacc_norm) <= 1.0

    # physics: the channel column accumulates far more than the edges. These
    # are log1p values, so compare the underlying counts, not the logs.
    cx = z.shape[1] // 2
    channel_cells = np.nanmean(np.expm1(flowacc_log[:, cx - 1:cx + 2]))
    edge_cells = np.nanmean(np.expm1(flowacc_log[:, 0:3]))
    assert channel_cells > edge_cells * 3.0


def test_flow_accumulation_respects_nodata(tmp_path):
    z, mask, meta = _channel_dem()
    mask = mask.copy(); mask[0:20, 0:20] = True
    flowacc_log, flowacc_norm = hydrology.flow_accumulation(
        z, mask, meta, str(tmp_path / "hydro2"))
    if flowacc_log is None:
        pytest.skip("WhiteboxTools unavailable in this environment")
    assert np.isnan(flowacc_log[mask]).all()
    assert (flowacc_norm[mask] == 0.0).all()
