"""Unit tests for terrain derivatives on synthetic DEMs with known geometry.

Assert directions and magnitudes within tolerance, never exact floats. The
grids here are metric (res = 1 m) so slope/aspect/TPI mirror the native-grid
math the real pipeline runs.
"""
import numpy as np
import pytest

from backend.pipeline import terrain

RES = 1.0


def _plane(shape=(200, 200), dz_dx=0.0, dz_dy=0.0, base=300.0):
    """Tilted plane. dz_dx rises toward +x (east), dz_dy toward +y (down/south
    in array-row order). Returns a float32 elevation grid."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    return (base + dz_dx * xx * RES + dz_dy * yy * RES).astype("float32")


def _gaussian(shape=(200, 200), amp=10.0, sigma=25.0, base=300.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    g = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return (base + g).astype("float32")


def _center(a, m=30):
    """Interior block, away from edge filter artefacts."""
    return a[m:-m, m:-m]


# ---------------------------------------------------------------- slope
def test_flat_plane_zero_slope():
    z = _plane(dz_dx=0.0, dz_dy=0.0)
    slope_deg, slope_pct, _ = terrain.slope_aspect(z, RES)
    assert np.allclose(_center(slope_deg), 0.0, atol=1e-4)
    assert np.allclose(_center(slope_pct), 0.0, atol=1e-4)


def test_tilted_plane_known_slope():
    # 10% grade toward east: rise 0.10 m per 1 m run -> atan(0.10) = 5.71 deg.
    z = _plane(dz_dx=0.10, dz_dy=0.0)
    slope_deg, slope_pct, _ = terrain.slope_aspect(z, RES)
    assert np.allclose(_center(slope_deg), np.degrees(np.arctan(0.10)), atol=0.1)
    assert np.allclose(_center(slope_pct), 10.0, atol=0.2)


# ---------------------------------------------------------------- aspect
# Convention (CLAUDE.md): 0 = N, 90 = E, 180 = S, 270 = W. Aspect points
# downhill. A plane rising toward +x (east) faces its low side west -> ~270.
def test_aspect_faces_west_when_rising_east():
    z = _plane(dz_dx=0.10, dz_dy=0.0)
    _, _, aspect = terrain.slope_aspect(z, RES)
    a = _center(aspect)
    assert abs(float(np.median(a)) - 270.0) < 5.0


def test_aspect_faces_north_when_rising_south():
    # rising toward +y (south, increasing row) -> downhill faces north (~0/360).
    z = _plane(dz_dx=0.0, dz_dy=0.10)
    _, _, aspect = terrain.slope_aspect(z, RES)
    a = _center(aspect)
    # wrap-safe distance to 0/360
    med = float(np.median(a))
    d = min(med, 360.0 - med)
    assert d < 5.0


# ---------------------------------------------------------------- TPI / pooling
def test_pit_has_negative_tpi_and_pooling():
    z = _gaussian(amp=-12.0, sigma=25.0)  # a pit (dips below the plane)
    t = terrain.tpi(z, RES, window_m=100)
    ny, nx = z.shape
    assert t[ny // 2, nx // 2] < -0.5  # center sits below its surroundings
    mask = np.zeros_like(z, dtype=bool)
    stats = terrain.pooling_stats(t, mask)
    assert stats["pct_lt_-0.5"] > 0.0


def test_bump_has_positive_tpi_and_no_pooling():
    z = _gaussian(amp=12.0, sigma=25.0)  # a bump (rises above the plane)
    t = terrain.tpi(z, RES, window_m=100)
    ny, nx = z.shape
    assert t[ny // 2, nx // 2] > 0.5  # center sits above its surroundings
    # center of a pure bump should register little to no pooling
    mask = np.zeros_like(z, dtype=bool)
    stats = terrain.pooling_stats(t, mask)
    assert stats["pct_lt_-1.0"] < 20.0


# ---------------------------------------------------------------- masking
def test_nodata_does_not_poison_interior():
    """A NaN patch far from the interior block must not corrupt slope there."""
    z = _plane(dz_dx=0.10, dz_dy=0.0)
    z[0:10, 0:10] = np.nan
    slope_deg, _, _ = terrain.slope_aspect(z, RES)
    assert np.allclose(_center(slope_deg), np.degrees(np.arctan(0.10)), atol=0.2)
    assert np.isfinite(_center(slope_deg)).all()


# ---------------------------------------------------------------- stats shapes
def test_aspect_distribution_sums_reasonably():
    z = _plane(dz_dx=0.10, dz_dy=0.0)
    slope_deg, _, aspect = terrain.slope_aspect(z, RES)
    mask = np.zeros_like(z, dtype=bool)
    dist = terrain.aspect_distribution(aspect, slope_deg, mask)
    assert set(dist) == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    # a uniform west-facing plane should concentrate in W
    assert dist["W"] == max(dist.values())
    assert abs(sum(dist.values()) - 100.0) < 1.5


def test_plant_above_line_none_on_low_relief():
    z = _plane(dz_dx=0.001, dz_dy=0.0)  # a few cm of relief across the tile
    t = terrain.tpi(z, RES)
    mask = np.zeros_like(z, dtype=bool)
    assert terrain.plant_above_line(z, t, mask) is None


def test_elevation_stats_relief():
    z = _plane(dz_dx=0.5, dz_dy=0.0, base=300.0)  # 0.5 m/px over ~200 px
    mask = np.zeros_like(z, dtype=bool)
    es = terrain.elevation_stats(z, mask)
    assert es["relief_m"] == pytest.approx(0.5 * (z.shape[1] - 1), abs=1.0)
