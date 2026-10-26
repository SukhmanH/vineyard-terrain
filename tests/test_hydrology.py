"""Phase-4 cold-air routing tests. These exercise the real WhiteboxTools
binary (downloaded on first use). If Whitebox cannot run, the module returns
None and the tests skip rather than fail, since the pipeline is designed to
degrade without it."""
import numpy as np
import pytest
from rasterio.transform import from_origin

from backend.pipeline import hydrology


def _meta(ny, nx):
    return {
        "crs": "EPSG:2955",
        "transform": from_origin(300000.0, 5500000.0, 1.0, 1.0),
        "res": 1.0, "width": nx, "height": ny, "nodata": -32767.0,
    }


def _channel_dem(ny=120, nx=120):
    """A plane draining north with a single N-S convergence channel."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    z = 300.0 + 0.05 * yy
    z -= 6.0 * np.exp(-(((xx - nx / 2) ** 2) / (2 * 4.0 ** 2)))
    return z.astype("float32"), np.zeros((ny, nx), dtype=bool), _meta(ny, nx)


def _route(z, mask, meta, tmp_path, name):
    r = hydrology.cold_air_routing(z, mask, meta, str(tmp_path / name))
    if r is None:
        pytest.skip("WhiteboxTools unavailable in this environment")
    return r


def test_routing_finds_channel(tmp_path):
    z, mask, meta = _channel_dem()
    r = _route(z, mask, meta, tmp_path, "hydro")

    # contract: finite on valid ground, log layer consistent with sca
    for k in ("sca", "sink_depth", "flowacc_log"):
        assert r[k].shape == z.shape
        assert np.isfinite(r[k][~mask]).all(), k
    assert np.allclose(r["flowacc_log"], np.log1p(r["sca"]))

    # physics: the channel column carries far more upslope area than the edges
    cx = z.shape[1] // 2
    channel = np.nanmean(r["sca"][:, cx - 1:cx + 2])
    edge = np.nanmean(r["sca"][:, 0:3])
    assert channel > edge * 3.0
    # an open channel on a plane holds no ponded cold air
    assert np.nanmax(r["sink_depth"]) < 0.05


def test_sink_depth_measures_a_closed_hollow(tmp_path):
    ny = nx = 160
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    z = 300.0 + 0.02 * yy
    z -= 3.0 * np.exp(-(((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 12.0 ** 2)))
    r = _route(z.astype("float32"), np.zeros((ny, nx), dtype=bool),
               _meta(ny, nx), tmp_path, "pit")
    assert 1.5 < r["sink_depth"][80, 80] < 3.1      # ponds most of the 3 m pit
    assert r["sink_depth"][10:30, 10:30].max() < 0.01


def test_routing_respects_nodata(tmp_path):
    z, mask, meta = _channel_dem()
    mask = mask.copy()
    mask[0:20, 0:20] = True
    r = _route(z, mask, meta, tmp_path, "hydro2")
    for k in ("sca", "sink_depth", "flowacc_log"):
        assert np.isnan(r[k][mask]).all(), k
        assert np.isfinite(r[k][~mask]).all(), k
