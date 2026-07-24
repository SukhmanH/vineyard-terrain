"""End-to-end pipeline test: write a synthetic GeoTIFF, run the orchestrator,
assert the full output contract is produced and stats.json parses.
"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from backend.pipeline import run

NODATA = -32767.0


def _write_synthetic_dtm(path, ny=300, nx=300, res=1.0):
    """A tilted plane with a pit, a NoData corner, in a real UTM 11N CRS.
    Named *_dtm.tif so the pipeline treats it as bare earth (no canopy caveat)."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype("float32")
    z = 300.0 + 0.08 * xx + 0.02 * yy  # gentle east-and-south rise, ~90 m relief
    cy, cx = ny * 0.5, nx * 0.5
    z -= 8.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 20.0 ** 2)))
    z[0:40, 0:40] = NODATA  # partial-coverage corner, as real tiles have
    # NAD83(CSRS) UTM 11N realization, like the real LidarBC tiles (EPSG:2955)
    transform = from_origin(300000.0, 5500000.0, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs="EPSG:2955", transform=transform, nodata=NODATA,
    ) as ds:
        ds.write(z.astype("float32"), 1)


def test_run_produces_output_contract(tmp_path):
    tif = tmp_path / "bc_082test_1_1_2_xl1m_utm11_990101_dtm.tif"
    _write_synthetic_dtm(str(tif))
    out_dir = tmp_path / "job1"

    stats = run.run(str(tif), str(out_dir))

    # ---- files present per the output contract
    assert (out_dir / "bounds.json").is_file()
    assert (out_dir / "stats.json").is_file()
    assert (out_dir / "meta.json").is_file()
    # phases 1-3 + phase-5 layers always render (no external dependency)
    for layer in ("elevation", "hillshade", "slope", "aspect", "tpi",
                  "frost", "suitability"):
        assert (out_dir / "layers" / f"{layer}.png").is_file(), layer
    # the Whitebox temp dir must not survive
    assert not (out_dir / "_hydro").exists()

    # ---- bounds.json is WGS84 lat/lon in the right hemisphere/quadrant
    bounds = json.loads((out_dir / "bounds.json").read_text())
    assert set(bounds) == {"south", "west", "north", "east"}
    assert 48.0 < bounds["south"] < 51.0 and bounds["north"] > bounds["south"]
    assert -121.0 < bounds["west"] < -118.0 and bounds["east"] > bounds["west"]

    # ---- stats.json parses and honors the contract
    parsed = json.loads((out_dir / "stats.json").read_text())
    assert parsed == stats  # returned dict matches the file
    assert parsed["coverage"]["valid_ha"] > 0
    assert parsed["coverage"]["nodata_pct"] > 0  # the corner we blanked
    assert parsed["elevation"]["relief_m"] > 0
    assert parsed["flags"]["is_dsm"] is False       # *_dtm.tif
    assert parsed["flags"]["canopy_caveat"] is False

    # ---- phase 5: frost + suitability present and self-consistent
    assert 0.0 <= parsed["frost"]["pct_high_risk"] <= 100.0
    ha = parsed["suitability_areas_ha"]
    assert set(ha) == {"excellent", "good", "marginal", "avoid"}
    # class areas should sum to roughly the valid area
    assert sum(ha.values()) == pytest.approx(
        parsed["coverage"]["valid_ha"], rel=0.02)
    assert set(parsed["suitability_areas_acres"]) == set(ha)

    # ---- phase 4: drainage flag reflects whether Whitebox produced flowacc
    if parsed["flags"].get("has_drainage"):
        assert (out_dir / "layers" / "flowacc.png").is_file()

    # ---- meta.json contract
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["filename"].endswith("_dtm.tif")
    assert meta["is_dsm"] is False
    assert "2955" in meta["crs"] or "NAD83" in meta["crs"]
    assert meta["res_m"] == 1.0


def test_unknown_suffix_flags_canopy_caveat(tmp_path):
    tif = tmp_path / "bc_082e999_1_1_2_xl1m_utm11_170606.tif"  # no dsm/dtm
    _write_synthetic_dtm(str(tif))
    out_dir = tmp_path / "job2"
    stats = run.run(str(tif), str(out_dir))
    assert stats["flags"]["is_dsm"] is True
    assert stats["flags"]["canopy_caveat"] is True
