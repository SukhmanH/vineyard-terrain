"""Unit tests for tile loading, model-type detection, and coverage stats."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.pipeline import loader

NODATA = -32767.0


def test_detect_model_type():
    assert loader.detect_model_type("bc_082e053_xl1m_utm11_170606_dtm.tif") == (False, False)
    assert loader.detect_model_type("bc_082e053_xl1m_utm11_170606_dsm.tif") == (True, True)
    # no suffix -> treat as possible surface model, caveat on
    assert loader.detect_model_type("bc_082e053_1_1_2_xl1m_utm11_170606.tif") == (True, True)
    # path components must not fool it; only the basename matters
    assert loader.detect_model_type("/some/dtm_folder/tile_dsm.tif") == (True, True)


def test_load_dem_masks_nodata(tmp_path):
    tif = tmp_path / "t.tif"
    z = np.full((50, 50), 310.0, dtype="float32")
    z[0:5, 0:5] = NODATA
    transform = from_origin(300000.0, 5500000.0, 1.0, 1.0)
    with rasterio.open(tif, "w", driver="GTiff", height=50, width=50, count=1,
                       dtype="float32", crs="EPSG:2955", transform=transform,
                       nodata=NODATA) as ds:
        ds.write(z, 1)

    arr, mask, meta = loader.load_dem(str(tif))
    assert mask.sum() == 25                      # the 5x5 blanked corner
    assert np.isnan(arr[mask]).all()             # NoData became NaN
    assert np.isfinite(arr[~mask]).all()         # valid cells intact
    assert meta["res"] == 1.0
    assert meta["nodata"] == NODATA


def test_coverage_stats():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :50] = True  # half NoData
    cov = loader.coverage_stats(mask, res=1.0)
    assert cov["nodata_pct"] == 50.0
    assert cov["valid_ha"] == round(5000 * 1.0 * 1.0 / 10000.0, 1)  # 0.5 ha
    assert cov["valid_acres"] == round(cov["valid_ha"] * 2.471, 1)
