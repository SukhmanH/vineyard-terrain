"""Per-parcel frost metrics: the analysis runs on the parcel's AOI (parcel + buffer) in
context, and this summarizes it inside the parcel.

Everything is on the fixed physical scales of frost model v2 (scoring.py). Nothing is
normalized by the job, the AOI or the set of listings: a frost score of 80 means the same
thing on every parcel, which is what makes listings comparable.

All of it is RELATIVE terrain risk of spring and fall radiative frost: clear calm nights,
when cold air drains downslope and pools. It is not a temperature, it never says a site
will not frost, and it says nothing about a multi-day arctic freeze like January 2024.
"""
import numpy as np
from scipy import ndimage

from . import scoring

ZONAL_VERSION = 1
MODEL_VERSION = f"{scoring.MODEL}+z{ZONAL_VERSION}"
LABEL = "spring/fall frost risk (terrain)"
SQM_PER_ACRE = 4046.8564224

FROST_SAFE_MAX = 40.0          # frost-safe ground: frost risk below this (owner, Oct 2026)
LOW_RISK_MAX = 30.0            # pct_low_risk counts frost risk below this
HIGH_RISK = scoring.HIGH_RISK  # pct_high_risk counts frost risk at or above this (60)
SAFE_CONNECTIVITY = ndimage.generate_binary_structure(2, 1)   # 4-neighbour: rows run
                                                              # through edges, not corners
LOW_COVERAGE_PCT = 80.0        # below: flag low_coverage, confidence low
FULL_COVERAGE_PCT = 95.0       # below: confidence drops one level
WEAK_FLOOR_CONFIDENCE = 0.5    # a block-detected floor below this drops one level
CANOPY_MIN_M = 3.0             # DSM minus DEM above this counts as canopy
EDGE_RISE_MIN = 0.01           # the surface must rise 1 % toward the missing side
EDGE_FLUX_MIN = 0.5            # half an edge cell of inflow reaching a parcel cell counts
EDGE_INFLOW_MAX_PCT = 5.0      # more parcel cells reached than this: widen once, then flag


# ------------------------------------------------------------- buffer adequacy
def inflow_cells(zs, valid, water, res):
    """Land cells where cold air from beyond the analysed ground would enter: next to the
    AOI rim or to NoData (not water, which is an outlet), with the field surface rising
    toward that side. Returns (rim, nodata) boolean grids."""
    land = valid & ~water
    # nearest-cell fill continues each edge outward, so a gap does not bend the gradient
    gy, gx = np.gradient(scoring._nearest_fill(np.where(valid, zs, 0.0), ~valid), res)
    missing = np.pad(~valid & ~water, 1, constant_values=False)
    outside = np.pad(np.zeros(zs.shape, bool), 1, constant_values=True)
    rim = np.zeros(zs.shape, bool)
    hole = np.zeros(zs.shape, bool)
    h, w = zs.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb_out = outside[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        nb_missing = missing[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        rises = gy * dr + gx * dc > EDGE_RISE_MIN
        rim |= land & nb_out & rises
        hole |= land & nb_missing & rises
    return rim, hole


def reached_pct(flux, sel):
    """Percent of sel cells that edge-origin cold air reaches (None when not computed)."""
    if flux is None or not sel.any():
        return None
    return round(100.0 * float((np.nan_to_num(flux[sel]) > EDGE_FLUX_MIN).mean()), 1)


# -------------------------------------------------------------------- metrics
def largest_component(cells):
    """Size in cells of the largest 4-connected group of True cells."""
    if not cells.any():
        return 0
    lab, n = ndimage.label(cells, structure=SAFE_CONNECTIVITY)
    return int(np.bincount(lab.ravel())[1:].max()) if n else 0


def confidence_level(parcel_confidence, coverage_pct, *, mixed=False, weak_floor=False,
                     buffer_inadequate=False):
    """high/medium/low. Low coverage or an unconfirmed parcel is low outright; each other
    doubt costs one level."""
    if parcel_confidence == "low" or coverage_pct < LOW_COVERAGE_PCT:
        return "low"
    level = {"high": 2, "medium": 1}.get(parcel_confidence, 0)
    level -= coverage_pct < FULL_COVERAGE_PCT
    level -= bool(mixed)
    level -= bool(weak_floor)
    level -= bool(buffer_inadequate)
    return ("low", "medium", "high")[max(int(level), 0)]


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def parcel_metrics(detail, zs, inside, valid, res, *, parcel_confidence="high",
                   parcel_flags=(), plant_above_m=None, canopy=None, edge_inflow_pct=None,
                   nodata_inflow_pct=None, widened=False, mix=None, aoi_resampled_m=None):
    """The `parcel` block of stats.json.

    detail: scoring.score_detail() over the AOI. zs: field-scale surface. inside: the parcel
    on the AOI grid. valid: bare earth present. canopy: DSM minus DEM (NaN where unknown)
    or None when no DSM twin exists.
    """
    cell_ac = res * res / SQM_PER_ACRE
    frost, land, info = detail["frost"], detail["land"], detail["info"]
    n_parcel = int(inside.sum())
    coverage = 100.0 * float((inside & valid).sum()) / n_parcel if n_parcel else 0.0
    scorable = inside & land
    plantable = scorable & (detail["slope_factor"] > 0)
    safe = plantable & (frost < FROST_SAFE_MAX)
    fv = frost[scorable]
    flags = list(parcel_flags)

    if fv.size:
        risk_mean, risk_p90 = float(fv.mean()), float(np.percentile(fv, 90))
        pct_high = 100.0 * float((fv >= HIGH_RISK).mean())
        pct_low = 100.0 * float((fv < LOW_RISK_MAX).mean())
        above = (zs - info["floor"])[scorable]
        height = float(np.median(above))
        drivers = scoring.drivers_pct(detail["comp"], scorable)
        sv, suit_ac, lower = detail["suit"][scorable], {}, np.inf
        for name, cut in scoring.CLASSES:      # counted in cells, not rounded hectares
            suit_ac[name] = _r(((sv >= cut) & (sv < lower)).sum() * cell_ac, 2)
            lower = cut
    else:
        risk_mean = risk_p90 = pct_high = pct_low = height = None
        drivers = {k: 0.0 for k in scoring.WEIGHTS}
        suit_ac = {name: 0.0 for name, _ in scoring.CLASSES}

    frost_score = None
    if plantable.any():
        frost_score = 100.0 - float(frost[plantable].mean())
    else:
        flags.append("no_plantable_land")

    canopy_ac = None
    if canopy is not None:
        canopy_ac = float((scorable & (np.nan_to_num(canopy) > CANOPY_MIN_M)).sum()) * cell_ac
        if canopy_ac > 0:
            flags.append("canopy_present")

    mixed = bool(mix) and not mix.get("refused")
    buffer_inadequate = edge_inflow_pct is not None and edge_inflow_pct > EDGE_INFLOW_MAX_PCT
    weak_floor = (info["floor_source"] == "block"
                  and info["floor_confidence"] < WEAK_FLOOR_CONFIDENCE)
    if coverage < LOW_COVERAGE_PCT:
        flags.append("low_coverage")
    if mixed:
        flags.append("mixed_deliveries")
    if mix and mix.get("refused"):
        flags.append("delivery_mix_refused")
    if widened:
        flags.append("buffer_widened")
    if buffer_inadequate:
        flags.append("buffer_inadequate")
    if nodata_inflow_pct is not None and nodata_inflow_pct > EDGE_INFLOW_MAX_PCT:
        flags.append("upslope_nodata")
    if info["floor_source"] == "block":
        flags.append("floor_from_block")
    if weak_floor:
        flags.append("weak_floor")
    if aoi_resampled_m:
        flags.append("aoi_resampled_2m")
    water_ac = float((inside & detail["water"]).sum()) * cell_ac
    if water_ac > 0:
        flags.append("water_in_parcel")

    return {
        "label": LABEL,
        "model_version": MODEL_VERSION,
        "coverage_pct": _r(coverage),
        "scorable_acres": _r(scorable.sum() * cell_ac, 2),
        "plantable_acres": _r(plantable.sum() * cell_ac, 2),
        "water_acres": _r(water_ac, 2),
        "frost_score": _r(frost_score),
        "frost_risk_mean": _r(risk_mean),
        "frost_risk_p90": _r(risk_p90),
        "pct_high_risk": _r(pct_high),
        "pct_low_risk": _r(pct_low),
        "frost_safe_acres": _r(safe.sum() * cell_ac, 2),
        "largest_contiguous_safe_acres": _r(largest_component(safe) * cell_ac, 2),
        "drivers_pct": drivers,
        "height_above_floor_median_m": _r(height),
        "floor_m": _r(info["floor_m"]),
        "floor_source": info["floor_source"],
        "floor_confidence": _r(info["floor_confidence"], 2),
        "plant_above_m": plant_above_m,
        "suitability_acres": suit_ac,
        "canopy_over_3m_acres": _r(canopy_ac, 2),
        "edge_inflow_pct": edge_inflow_pct,
        "nodata_inflow_pct": nodata_inflow_pct,
        "confidence": confidence_level(parcel_confidence, coverage, mixed=mixed,
                                       weak_floor=weak_floor,
                                       buffer_inadequate=buffer_inadequate),
        "flags": flags,
        "thresholds": {"frost_safe_max": FROST_SAFE_MAX, "low_risk_max": LOW_RISK_MAX,
                       "high_risk": HIGH_RISK, "canopy_min_m": CANOPY_MIN_M},
    }
