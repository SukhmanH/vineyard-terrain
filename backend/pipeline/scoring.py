"""Phase 5. Frost risk and vineyard suitability composites.

v1 heuristic, parameters fixed in CLAUDE.md, expected to be tuned against the
owner's ESP32 sensor network once it is collecting. Everything here is RELATIVE
terrain risk on the current job's valid area; nothing is a temperature and no
site is ever called frost free.

Frost risk combines three drivers of nocturnal cold-air pooling:
  pooling    - how far a cell sits below its surroundings (from TPI)
  drainage   - cold-air convergence from flow accumulation (phase 4; 0 without)
  low_ground - being low in the tile's elevation range
Suitability then rewards low frost risk, a warm S-SW aspect, and a workable,
self-draining slope.
"""
import numpy as np

ACRES_PER_HA = 2.471
ASPECT_OPTIMUM_DEG = 190.0  # S-SW, the warm optimum near 49 N


def _percentile_rank(values):
    """Rank each value in 0..1 by its position in the sorted distribution.
    Ties share the average rank. Operates on a 1-D array of valid cells."""
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.arange(n, dtype="float64")
    # average-rank ties so plateaus do not bias low_ground
    sv = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            avg = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks / max(n - 1, 1)


def _slope_factor(slope_pct):
    """Piecewise workability/self-drainage factor on slope percent.
    <2: 0.85, 2..17: 1.0, 17..30: 1.0->0.6, 30..45: 0.6->0.3, >45: 0.0."""
    sp = slope_pct
    f = np.empty_like(sp, dtype="float32")
    f[:] = 0.85                                   # flat default
    f[(sp >= 2) & (sp <= 17)] = 1.0
    m = (sp > 17) & (sp <= 30)
    f[m] = 1.0 + (sp[m] - 17.0) * (0.6 - 1.0) / (30.0 - 17.0)
    m = (sp > 30) & (sp <= 45)
    f[m] = 0.6 + (sp[m] - 30.0) * (0.3 - 0.6) / (45.0 - 30.0)
    f[sp > 45] = 0.0
    return f


def frost_and_suitability(z, tpi, slope_deg, slope_pct, aspect_deg,
                          flowacc_norm, mask, res):
    """Return (frost_pct, suitability_pct, stats). Both grids are float32 with
    NaN outside valid ground; stats carries the phase-5 keys for stats.json."""
    valid = ~mask
    shape = z.shape

    frost = np.full(shape, np.nan, dtype="float32")
    suit = np.full(shape, np.nan, dtype="float32")

    zv = z[valid]
    tv = tpi[valid]
    if zv.size == 0:
        stats = {
            "frost": {"pct_high_risk": 0.0},
            "suitability_areas_ha": {"excellent": 0.0, "good": 0.0,
                                     "marginal": 0.0, "avoid": 0.0},
            "suitability_areas_acres": {"excellent": 0.0, "good": 0.0,
                                        "marginal": 0.0, "avoid": 0.0},
        }
        return frost, suit, stats

    # ---- frost risk components (all 0..1 over valid cells) ----------------
    pooling_raw = np.clip(-tv, 0.0, None)
    p95 = float(np.percentile(pooling_raw, 95))
    pooling = pooling_raw / p95 if p95 > 0 else np.zeros_like(pooling_raw)
    pooling = np.clip(pooling, 0.0, 1.0)

    if flowacc_norm is not None:
        drainage = np.clip(flowacc_norm[valid].astype("float64"), 0.0, 1.0)
    else:
        drainage = np.zeros_like(pooling)

    low_ground = 1.0 - _percentile_rank(zv.astype("float64"))

    frost_v = 100.0 * (0.45 * pooling + 0.35 * drainage + 0.20 * low_ground)
    frost_v = np.clip(frost_v, 0.0, 100.0)

    # ---- suitability -----------------------------------------------------
    asp = aspect_deg[valid]
    sd = slope_deg[valid]
    sp = slope_pct[valid]
    aspect_score = np.clip(
        np.cos(np.radians(asp - ASPECT_OPTIMUM_DEG)), 0.0, 1.0)
    aspect_score[sd < 2.0] = 0.5          # aspect is meaningless on flat ground
    slope_factor = _slope_factor(sp)

    suit_v = np.clip(
        (100.0 - frost_v) * (0.6 + 0.4 * aspect_score) * slope_factor,
        0.0, 100.0)

    frost[valid] = frost_v.astype("float32")
    suit[valid] = suit_v.astype("float32")

    # ---- area accounting -------------------------------------------------
    cell_ha = (res * res) / 10000.0
    classes = {
        "excellent": suit_v >= 70.0,
        "good": (suit_v >= 50.0) & (suit_v < 70.0),
        "marginal": (suit_v >= 30.0) & (suit_v < 50.0),
        "avoid": suit_v < 30.0,
    }
    areas_ha = {k: round(float(sel.sum()) * cell_ha, 1)
                for k, sel in classes.items()}
    areas_acres = {k: round(v * ACRES_PER_HA, 1) for k, v in areas_ha.items()}

    stats = {
        "frost": {"pct_high_risk": round(
            100.0 * float(np.mean(frost_v >= 60.0)), 1)},
        "suitability_areas_ha": areas_ha,
        "suitability_areas_acres": areas_acres,
    }
    return frost, suit, stats
