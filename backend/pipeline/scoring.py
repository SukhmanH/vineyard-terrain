"""Phase 5. Frost risk and vineyard suitability composites (model v2).

Everything here is RELATIVE terrain risk. Nothing is a temperature and no site
is ever called frost free. The owner's ESP32 sensor network is the ground
truth these constants get tuned against.

Why v2: v1 scaled every component by the job's own percentiles, so every block
came out with about the same share of high-risk ground (7 to 10 pct on the
Osoyoos valley floor and on the Penticton bench alike) and the valley floor
itself, the coldest ground in the region, scored low because TPI is ~0 on a
broad flat. v2 scores each mechanism on a fixed physical scale (metres of
ponding, metres above the valley floor, a topographic index, sun relative to
flat ground), so a floor scores colder than a bench in whichever block it sits.

Frost risk. On clear calm nights cold air drains downslope and collects. Four
mechanisms put a cell inside that cold air, each scored 0..1:
  closed_hollow  depth below the spill point of a closed depression. Cold air
                 cannot drain out until it overtops the rim; a road embankment
                 or berm across a slope dams it the same way.
  valley_floor   height above the valley floor. The regional cold pool fills
                 from the bottom, so the lowest ground is coldest and the
                 thermal belt sits tens of metres up the valley side.
  cold_air_flow  topographic convergence index ln(a / tan b): plenty of
                 upslope ground draining in (a, specific contributing area)
                 onto ground too gentle (b) to keep the cold air moving.
  low_spot       sitting below the surrounding 200 m (negative TPI): toe
                 slopes, swales and open hollows.
They combine as independent causes, risk = 1 - prod(1 - w_i c_i), so one
strong mechanism is enough for high risk, as a deep hollow on a bench is.

Suitability = (100 - frost) x heat x slope.
  heat   growing-season (Apr-Oct) clear-sky sun on the actual slope and
         aspect at the block's latitude, relative to flat ground, with warm
         afternoon sun counting more than cool early-morning sun.
  slope  workability on field-scale slope percent.

Slope, aspect and the routing surface are computed at field scale (3 m
Gaussian) so vine rows, ruts and 1 m LiDAR speckle do not drive the scores.
Hydro-flattened water (lakes, oxbow ponds) is excluded from scores and areas.
"""
import functools

import numpy as np
from scipy import ndimage

ACRES_PER_HA = 2.471
MODEL = "v2"

# ---- field scale -----------------------------------------------------------
FIELD_SIGMA_M = 3.0       # Gaussian for slope, aspect and the routing surface

# ---- water: hydro-flattened lakes and ponds ----------------------------------
WATER_WINDOW = 5          # cells; LiDAR ground is never exactly level this wide
WATER_MIN_HA = 0.1

# ---- frost mechanisms: fixed physical scales, never per-block percentiles ----
SINK_MIN_M = 0.1          # closed hollows shallower than this are LiDAR noise
SINK_SCALE_M = 1.0        # ~1 m of ponded cold air reaches bud height
FLOOR_DECAY_M = 40.0      # e-folding height of the valley cold pool
FLOOR_BAND_M = 5.0        # lowest ground examined to recognise a valley floor
FLOOR_FLAT_PCT = 5.0      # a valley floor is flatter than this
FLOOR_FLAT_HA = 4.0       # ... and at least this much of it is in the block
FLOOR_RELIEF_M = 30.0     # ground must rise this far above it (valley sides)

# South Okanagan valley floor: lake and river levels at points along the
# valley axis, north to south (lat, lon, metres). A block within
# VALLEY_AXIS_KM of the axis takes its cold-pool floor from here, because a
# bench block cannot see the floor below it: over the full Penticton bench
# tile the lowest flats sit at 419 m, 77 m above Okanagan Lake, and would
# otherwise be scored as the valley floor. Checked against the lowest ground
# of real LidarBC floor tiles: Okanagan Lake 343.5 (June 2017 flood stage),
# below Okanagan Falls 328.5, Oliver 296.0, Golden Mile 288.5, north
# Osoyoos 278.4. Elsewhere the floor is detected from the block itself.
VALLEY_FLOOR_AXIS = (
    (49.620, -119.620, 342.0),   # Okanagan Lake off Summerland and Naramata
    (49.500, -119.595, 342.0),   # Okanagan Lake, Penticton shore
    (49.460, -119.585, 338.0),   # Skaha Lake, north end
    (49.360, -119.570, 338.0),   # Skaha Lake, south end
    (49.335, -119.565, 329.5),   # Okanagan River below Okanagan Falls
    (49.290, -119.540, 327.0),   # Vaseux Lake
    (49.210, -119.545, 297.0),   # Oliver
    (49.165, -119.540, 290.0),   # Golden Mile and Black Sage
    (49.100, -119.530, 280.0),   # north of Osoyoos
    (49.040, -119.465, 277.5),   # Osoyoos Lake
    (48.950, -119.430, 277.5),   # Osoyoos Lake, south basin
)
VALLEY_AXIS_KM = 4.0
TCI_MID = 9.0             # ln(m): convergence index scoring 0.5
TCI_WIDTH = 1.0
TCI_MIN_TAN = 0.005       # floor on tan(slope) so flats stay finite
LOW_SPOT_WINDOW_M = 200
LOW_SPOT_SCALE_M = 2.0    # metres below the 200 m surroundings

WEIGHTS = {               # the most each mechanism can contribute on its own
    "closed_hollow": 0.85,
    "valley_floor": 0.75,
    "cold_air_flow": 0.50,
    "low_spot": 0.50,
}
HIGH_RISK = 60.0          # frost >= this counts as high risk in stats

# ---- heat: growing-season sun ----------------------------------------------
SEASON_DOY = (91, 304)    # Apr 1 to Oct 31, bud break to harvest
AFTERNOON_WEIGHT = 0.2    # 3 pm sun counts 1.2x, 6 am sun 0.86x
SUN_COOL = 0.75           # sun relative to flat that scores 0 (steep north)
SUN_WARM = 1.06           # ... that scores 1 (good S-SW slope)
DEFAULT_LAT = 49.3        # south Okanagan, used when the caller has none

CLASSES = (("excellent", 70.0), ("good", 50.0), ("marginal", 30.0),
           ("avoid", -1.0))


# ------------------------------------------------------------- field surface
# Grids are float32 throughout: a four-tile mosaic is 11 M cells and the API
# runs on a 2 GB machine.
def _nearest_fill(z, mask):
    """Fill NoData from the nearest valid cell. Unlike a median fill this
    continues the edge elevation outward, so filters and gradients near a
    data gap see no fake cliff or pit."""
    if not mask.any():
        return z.astype("float32")
    idx = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True)
    return z[tuple(idx)].astype("float32")


def field_surface(z, mask, res):
    """Field-scale surface shared by routing and scoring.

    Returns a dict of float32 grids (NaN at NoData except "z", which stays
    filled for Whitebox and filters): z, slope_deg, slope_pct, aspect_deg.
    """
    zs = ndimage.gaussian_filter(_nearest_fill(z, mask), FIELD_SIGMA_M / res,
                                 mode="nearest")
    gy, gx = np.gradient(zs, res)
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0
    grad = np.hypot(gx, gy)
    del gx, gy
    out = {
        "z": zs,
        "slope_deg": np.degrees(np.arctan(grad)),
        "slope_pct": 100.0 * grad,
        "aspect_deg": aspect,
    }
    for k in ("slope_deg", "slope_pct", "aspect_deg"):
        out[k][mask] = np.nan
    return out


def water_mask(z, mask, res):
    """Hydro-flattened water: connected areas of exactly constant elevation.
    LidarBC bare-earth DEMs flatten lakes and ponds; real ground always varies
    at LiDAR precision over a 5 m window."""
    hi = ndimage.maximum_filter(np.where(mask, -np.inf, z), WATER_WINDOW)
    flat = (hi == ndimage.minimum_filter(np.where(mask, np.inf, z),
                                         WATER_WINDOW)) & ~mask
    del hi
    if not flat.any():
        return flat
    lab, n = ndimage.label(flat)
    sizes = np.bincount(lab.ravel())
    keep = sizes * res * res / 10000.0 >= WATER_MIN_HA
    keep[0] = False
    core = keep[lab]
    # the window test misses the last cells before the shore; take the rest
    # of each water body's level surface
    level = ndimage.maximum_filter(np.where(core, z, -np.inf), WATER_WINDOW)
    return (core | (z == level)) & ~mask


def _plane_fit(zc, w, size):
    """Value at each cell of the least-squares plane through the weighted
    cells (weights w in 0..1) of the surrounding size x size window."""
    ny, nx = zc.shape
    # coordinates centred on the block keep the moment sums small
    x = (np.arange(nx, dtype="float64") - nx / 2.0)[None, :]
    y = (np.arange(ny, dtype="float64") - ny / 2.0)[:, None]

    def mean(a):  # window sum over weighted cells; the window area cancels
        return ndimage.uniform_filter(a * w, size, mode="constant")

    s0 = np.maximum(mean(1.0), 1e-12)
    mx, my, mz = mean(x) / s0, mean(y) / s0, mean(zc) / s0
    cxx = mean(x * x) / s0 - mx * mx
    cyy = mean(y * y) / s0 - my * my
    cxy = mean(x * y) / s0 - mx * my
    cxz = mean(x * zc) / s0 - mx * mz
    cyz = mean(y * zc) / s0 - my * mz
    det = cxx * cyy - cxy * cxy
    ok = det > 1e-6 * np.maximum(cxx * cyy, 1e-12)
    safe = np.where(ok, det, 1.0)
    b = np.where(ok, (cxz * cyy - cyz * cxy) / safe, 0.0)   # dz/dx per cell
    c = np.where(ok, (cyz * cxx - cxz * cxy) / safe, 0.0)   # dz/dy per cell
    return mz + b * (x - mx) + c * (y - my)


def plane_tpi(z, valid, size, step=1):
    """Topographic position: metres above (+) or below (-) the plane fitted
    through the valid cells of the surrounding size x size window.

    In a full window the fitted plane passes through the window mean at the
    centre, so this equals classic TPI (z minus the window mean). Where the
    window is cut by the block edge or a NoData gap, a plain mean is dragged
    toward the ground that remains, so the downhill edge of a uniform slope
    reads as a hollow; the plane carries the local slope across the gap and
    a uniform slope reads 0 right to its edge.

    step > 1 fits the planes on a grid step cells coarse and interpolates
    them back. A 200 m plane is smooth at that spacing, and the fit then
    needs 1/step^2 of the memory.
    """
    ny, nx = z.shape
    zmean = float(z[valid].mean()) if valid.any() else 0.0
    zc = np.where(valid, z - zmean, 0.0).astype("float32")
    if step <= 1:
        plane = _plane_fit(zc.astype("float64"), valid.astype("float64"), size)
        return (zc - plane.astype("float32")).astype("float32")
    pad = ((0, (-ny) % step), (0, (-nx) % step))
    cy, cx = (ny + pad[0][1]) // step, (nx + pad[1][1]) // step
    zsum = np.pad(zc, pad).reshape(cy, step, cx, step).sum(axis=(1, 3),
                                                            dtype="float64")
    wsum = np.pad(valid, pad).reshape(cy, step, cx, step).sum(
        axis=(1, 3), dtype="float64")
    zcoarse = np.where(wsum > 0, zsum / np.maximum(wsum, 1.0), 0.0)
    plane = _plane_fit(zcoarse, wsum / (step * step),
                       max(3, int(round(size / step)) | 1))
    # one extra coarse cell all round, extrapolated linearly, so the edge
    # cells beyond the last coarse centre keep the slope instead of flattening
    plane = np.pad(plane, 1, mode="reflect", reflect_type="odd")
    plane = ndimage.zoom(plane.astype("float32"), step, order=1,
                         mode="nearest", grid_mode=True)
    return (zc - plane[step:step + ny, step:step + nx]).astype("float32")


# ------------------------------------------------------------------- heat
@functools.lru_cache(maxsize=8)
def _sun_table(lat_deg):
    """Growing-season sun on a slope relative to flat ground at lat_deg, as a
    table over slope (0..60 deg, 1 deg steps) x aspect (0..360 deg, 5 deg).

    Clear-sky beam with Meinel air-mass attenuation, plus isotropic diffuse
    and ground-reflected light, summed every 10 minutes on every 4th day of
    the season. Each time step is weighted 1 + AFTERNOON_WEIGHT x cos(hour
    angle - 45 deg), peaking at 3 pm, so sun landing in the warm afternoon
    counts more than early-morning sun. Shadows cast by distant ridges are not
    modelled; a 2 km tile does not contain the horizon.
    """
    s = np.radians(np.arange(0.0, 61.0, 1.0))[:, None, None]
    a = np.radians(np.arange(0.0, 365.0, 5.0))[None, :, None]
    ne, nn, nu = np.sin(s) * np.sin(a), np.sin(s) * np.cos(a), np.cos(s)
    phi = np.radians(lat_deg)
    omega = np.radians(np.arange(-180.0, 180.0, 2.5))     # hour angle, 10 min
    tw = 1.0 + AFTERNOON_WEIGHT * np.cos(omega - np.radians(45.0))
    tilted = np.zeros((s.shape[0], a.shape[1]))
    flat = 0.0
    for doy in range(SEASON_DOY[0], SEASON_DOY[1] + 1, 4):
        dec = np.radians(23.45) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
        up_ = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(omega)
        day = up_ > 0.01
        su = up_[day]
        se = -np.cos(dec) * np.sin(omega[day])                  # east
        sn = (np.cos(phi) * np.sin(dec)
              - np.sin(phi) * np.cos(dec) * np.cos(omega[day]))  # north
        alt = np.degrees(np.arcsin(su))
        am = 1.0 / (su + 0.50572 * (alt + 6.07995) ** -1.6364)  # Kasten-Young
        beam = 0.7 ** (am ** 0.678)                             # Meinel
        diff = 0.1 * beam
        w = tw[day]
        cos_i = np.clip(ne * se + nn * sn + nu * su, 0.0, None)
        tilted += ((beam * cos_i + diff * (1.0 + nu) / 2.0
                    + 0.2 * (beam * su + diff) * (1.0 - nu) / 2.0) * w).sum(-1)
        flat += float(((beam * su + diff) * w).sum())
    return tilted / flat


def sun_ratio(slope_deg, aspect_deg, lat_deg=DEFAULT_LAT):
    """Growing-season sun on each cell relative to flat ground (1.0 = flat).
    Varies smoothly with slope, so a gentle north slope is barely penalised
    and a steep one heavily, with no jump at any slope threshold."""
    table = _sun_table(round(float(lat_deg), 1))
    out = np.empty(np.shape(slope_deg), dtype="float32")
    for r in range(0, out.shape[0], 256):        # rows in chunks: bounded memory
        sd = np.nan_to_num(np.clip(slope_deg[r:r + 256], 0.0, 60.0), nan=0.0)
        ad = np.nan_to_num(np.mod(aspect_deg[r:r + 256], 360.0), nan=0.0)
        out[r:r + 256] = ndimage.map_coordinates(
            table, [sd.ravel(), ad.ravel() / 5.0], order=1,
            mode="nearest").reshape(sd.shape)
    return out


def heat_score(ratio):
    return np.clip((ratio - SUN_COOL) / (SUN_WARM - SUN_COOL), 0.0, 1.0)


# ------------------------------------------------------------------ slope
def _slope_factor(slope_pct):
    """Workability on field-scale slope percent. 0.85 on dead flat, full from
    3 to 17 pct, tapering to 0.6 at 30 and 0.3 at 45, 0 above 45 (not
    mechanizable). Continuous, so 1 m speckle cannot flip a cell."""
    f = np.interp(slope_pct, [0.0, 1.0, 3.0, 17.0, 30.0, 45.0],
                  [0.85, 0.85, 1.0, 1.0, 0.6, 0.3]).astype("float32")
    f[slope_pct > 45.0] = 0.0
    return f


# ------------------------------------------------------------------- frost
def _saturate(x, scale):
    """0 at x <= 0, rising to 1; 0.63 at x = scale."""
    return 1.0 - np.exp(-np.clip(x, 0.0, None) / scale)


def valley_floor_profile(lats, lons):
    """Regional valley-floor elevation (m) and distance from the valley axis
    (km) at each point, from VALLEY_FLOOR_AXIS. The floor is NaN farther than
    VALLEY_AXIS_KM from the axis, where the table says nothing."""
    kx = 111.32 * np.cos(np.radians(49.3))   # km per degree; flat earth is
    ky = 110.57                              # fine over a 70 km valley
    qx = np.asarray(lons, dtype="float64") * kx
    qy = np.asarray(lats, dtype="float64") * ky
    best_d = np.full(qx.shape, np.inf)
    best_f = np.full(qx.shape, np.nan)
    for (la0, lo0, f0), (la1, lo1, f1) in zip(VALLEY_FLOOR_AXIS[:-1],
                                              VALLEY_FLOOR_AXIS[1:]):
        ax, ay = lo0 * kx, la0 * ky
        dx, dy = (lo1 - lo0) * kx, (la1 - la0) * ky
        t = np.clip(((qx - ax) * dx + (qy - ay) * dy) / (dx * dx + dy * dy),
                    0.0, 1.0)
        d = np.hypot(qx - ax - t * dx, qy - ay - t * dy)
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_f = np.where(closer, f0 + t * (f1 - f0), best_f)
    return np.where(best_d <= VALLEY_AXIS_KM, best_f, np.nan), best_d


def block_geography(meta):
    """Latitude of the block centre (for the sun model) and the regional
    valley floor under each row (None when the block centre is off the
    mapped valley axis). The valley runs north-south and its floor falls 65 m
    from Penticton to Osoyoos, so a long mosaic gets a floor per row."""
    from rasterio.warp import transform
    t, h, w = meta["transform"], meta["height"], meta["width"]
    xc = t.c + t.a * w / 2.0
    ys = t.f + t.e * (np.arange(h) + 0.5)
    lons, lats = transform(meta["crs"], "EPSG:4326", [xc] * h, ys.tolist())
    lats, lons = np.asarray(lats), np.asarray(lons)
    floor, _ = valley_floor_profile(lats, lons)
    mid = h // 2
    rows = None
    if np.isfinite(floor[mid]):
        ok = np.isfinite(floor)
        idx = np.arange(h)
        rows = np.interp(idx, idx[ok], floor[ok])   # rows past the corridor
    return {"lat": float(lats[mid]), "floor_rows": rows}


def _floor_reference(zs, slope_pct, valid, res):
    """Locate the valley floor this block drains to, for a block off the
    mapped valley axis.

    The floor is the block's lowest ground (1st percentile), trusted only when
    a few hectares of it are flat and valley sides rise above it. A block
    whose lowest ground is a slope (the floor lies below the block) or that is
    flat with no sides (a bench top and a floor look the same) gets low
    confidence, and the valley-floor mechanism is scaled down by it.
    Real tiles: valley floors carry 30 to 130 ha of flat ground within 5 m of
    the base, the Penticton bench 0.4 ha. A creek ravine bottom (5 ha) passes
    as a floor, which overstates risk there rather than missing a real floor.
    Returns (floor_m, confidence 0..1).
    """
    zv = zs[valid]
    base = float(np.percentile(zv, 1))
    relief = float(np.percentile(zv, 99)) - base
    flat = valid & (zs < base + FLOOR_BAND_M) & (slope_pct < FLOOR_FLAT_PCT)
    flat_ha = float(flat.sum()) * res * res / 10000.0
    conf = (np.clip(flat_ha / FLOOR_FLAT_HA, 0.0, 1.0)
            * np.clip(relief / FLOOR_RELIEF_M, 0.0, 1.0))
    return base, float(conf)


def frost_components(z, mask, res, field=None, routing=None, water=None,
                     floor_rows=None):
    """Score each cold-air mechanism 0..1 on land cells.

    routing: dict from hydrology.cold_air_routing (sca, sink_depth) or None
    when Whitebox is unavailable; the two routing mechanisms then score 0.
    floor_rows: regional valley floor per row (block_geography), or None to
    detect the floor from the block itself.
    Returns (components dict of float32 grids, info dict).
    """
    if field is None:
        field = field_surface(z, mask, res)
    if water is None:
        water = water_mask(z, mask, res)
    valid = ~mask
    land = valid & ~water
    zs, sp = field["z"], field["slope_pct"]
    comp = {k: np.zeros(z.shape, dtype="float32") for k in WEIGHTS}

    # valley floor
    rows = valid.any(axis=1)
    if floor_rows is not None:
        floor = np.asarray(floor_rows, dtype="float64")[:, None]
        # the table is approximate: if the block's own lowest ground sits
        # below it, lower the profile to meet the block
        base = float(np.percentile(zs[valid], 1))
        floor = floor - max(0.0, float(floor[rows].min()) - base)
        conf, source = 1.0, "valley"
        floor_m = float(np.median(floor[rows]))
        floor = floor.astype("float32")
    else:
        # water counts as floor-level ground when locating it
        floor, conf = _floor_reference(zs, np.where(water, 0.0, sp), valid, res)
        source, floor_m = "block", floor
    comp["valley_floor"] = conf * np.exp(
        -np.clip(zs - floor, 0.0, None) / FLOOR_DECAY_M)

    # low spot: metres below the surrounding 200 m, valid cells only. The
    # planes are fitted on a 4 m grid; they are smooth at that spacing.
    w = max(3, int(round(LOW_SPOT_WINDOW_M / res)))
    step = max(1, int(round(4.0 / res)))
    comp["low_spot"] = _saturate(-plane_tpi(zs, valid, w, step),
                                 LOW_SPOT_SCALE_M)

    if routing is not None:
        depth = np.nan_to_num(routing["sink_depth"], nan=0.0)
        comp["closed_hollow"] = _saturate(depth - SINK_MIN_M, SINK_SCALE_M)
        del depth
        # a cold-air drainage flow is metres wide, not one cell: spread the
        # contributing area (a flux, so linear smoothing conserves it)
        sca = np.where(land, np.nan_to_num(routing["sca"], nan=0.0), 0.0)
        sig = FIELD_SIGMA_M / res
        sca = (ndimage.gaussian_filter(sca, sig)
               / np.maximum(ndimage.gaussian_filter(land.astype("float32"), sig),
                            1e-9))
        tan_b = np.maximum(np.nan_to_num(sp, nan=0.0) / 100.0, TCI_MIN_TAN)
        tci = np.log(np.maximum(sca, res) / tan_b)
        del sca, tan_b
        comp["cold_air_flow"] = 1.0 / (1.0 + np.exp(-(tci - TCI_MID) / TCI_WIDTH))

    for c in comp.values():
        c[~land] = np.nan
    info = {"floor_m": floor_m, "floor_confidence": conf,
            "floor_source": source, "land": land, "water": water}
    return comp, info


def combine_frost(comp):
    """Independent-causes combination, 0..100."""
    keep = np.ones_like(next(iter(comp.values())))
    for k, c in comp.items():
        keep *= 1.0 - WEIGHTS[k] * c
    return 100.0 * (1.0 - keep)


# ---------------------------------------------------------------- composite
def frost_and_suitability(z, mask, res, lat=None, field=None, routing=None,
                          water=None, floor_rows=None):
    """Return (frost_pct, suitability_pct, stats).

    field, routing and water are computed here when not supplied; run.py
    supplies them because routing needs the field surface and water first.
    lat and floor_rows come from block_geography(meta); without them the sun
    model uses DEFAULT_LAT and the floor is detected from the block.
    Both grids are float32 with NaN outside scored land (NoData and water);
    stats carries the phase-5 keys for stats.json.
    """
    shape = z.shape
    frost = np.full(shape, np.nan, dtype="float32")
    suit = np.full(shape, np.nan, dtype="float32")
    cell_ha = (res * res) / 10000.0

    if field is None:
        field = field_surface(z, mask, res)
    if water is None:
        water = water_mask(z, mask, res)
    land = ~mask & ~water
    if not land.any():
        return frost, suit, _empty_stats(float(water.sum()) * cell_ha)

    comp, info = frost_components(z, mask, res, field, routing, water,
                                  floor_rows)
    frost_g = combine_frost(comp)

    lat = DEFAULT_LAT if lat is None else lat
    ratio = sun_ratio(field["slope_deg"], field["aspect_deg"], lat)
    heat = heat_score(ratio)
    slope_f = _slope_factor(np.nan_to_num(field["slope_pct"], nan=0.0))
    suit_g = np.clip((100.0 - frost_g) * (0.6 + 0.4 * heat) * slope_f,
                     0.0, 100.0)

    frost[land] = frost_g[land]
    suit[land] = suit_g[land]
    fv, sv = frost_g[land], suit_g[land]

    # area accounting by suitability class
    areas_ha, lower = {}, np.inf
    for name, cut in CLASSES:
        areas_ha[name] = round(float(((sv >= cut) & (sv < lower)).sum())
                               * cell_ha, 1)
        lower = cut
    areas_acres = {k: round(v * ACRES_PER_HA, 1) for k, v in areas_ha.items()}

    # which mechanism dominates each high-risk cell
    high = fv >= HIGH_RISK
    names = list(WEIGHTS)
    drivers = {k: 0.0 for k in names}
    if high.any():
        contrib = np.stack([WEIGHTS[k] * comp[k][land][high] for k in names])
        top = np.argmax(contrib, axis=0)
        drivers = {k: round(100.0 * float(np.mean(top == i)), 1)
                   for i, k in enumerate(names)}

    stats = {
        "frost": {
            "model": MODEL,
            "pct_high_risk": round(100.0 * float(high.mean()), 1),
            "mean": round(float(fv.mean()), 1),
            "drivers_pct": drivers,
            "floor_m": round(info["floor_m"], 1),
            "floor_source": info["floor_source"],
            "floor_confidence": round(info["floor_confidence"], 2),
        },
        "heat": {"sun_vs_flat_mean": round(float(ratio[land].mean()), 3)},
        "suitability_areas_ha": areas_ha,
        "suitability_areas_acres": areas_acres,
        "water_ha": round(float(water.sum()) * cell_ha, 1),
    }
    return frost, suit, stats


def _empty_stats(water_ha=0.0):
    zero = {"excellent": 0.0, "good": 0.0, "marginal": 0.0, "avoid": 0.0}
    return {
        "frost": {"model": MODEL, "pct_high_risk": 0.0, "mean": 0.0,
                  "drivers_pct": {k: 0.0 for k in WEIGHTS},
                  "floor_m": None, "floor_source": None,
                  "floor_confidence": 0.0},
        "heat": {"sun_vs_flat_mean": None},
        "suitability_areas_ha": dict(zero),
        "suitability_areas_acres": dict(zero),
        "water_ha": round(water_ha, 1),
    }
