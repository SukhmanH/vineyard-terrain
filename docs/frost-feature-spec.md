# Add a frost-pocket / cold-air drainage feature to vineyard-terrain

## Inspiration / reference
- **Tool we're modelling this on:** CutSheetRX by Travis Senter (Senter Farms): https://www.senterfarms.com/programs/csrx/
  It's a free lidar map tool for drainage, elevation profiles and cut/fill. Open it to see the feel we want: a satellite map, analysis of whatever is on screen, near-instant loading, and a profile tool.
- **Author's announcement on X:** https://x.com/traviss22/status/2118375285917884898 (@traviss22)
- His site isn't open source (no repo or license found). Use it as a **reference for the user experience and the approach only**. Write our own code. Don't copy his scripts or branding. Our version is aimed at **frost**: cold-air drainage and frost pockets.

## Goal
Add a frost tool to this project. It reads bare-earth lidar for any area the user views and shows where cold air drains and pools on calm, clear (radiation-frost) nights. It must load fast: fetch elevation **only for the current view** from public services. Do not download lidar tiles or point clouds.

A working standalone prototype already exists (single HTML file, Leaflet + geotiff.js). Everything you need from it is below. **Fit it into this project's existing stack, structure and conventions. Don't paste in a second app.**

## Step 0: Read the project first
Before changing anything:
1. Read the README and package/config files, and work out the map library, framework and how elevation is already handled (if at all).
2. Reuse existing map, layer and UI code where possible.
3. Briefly tell me your integration plan before making big structural changes.

## Elevation data (no API keys, CORS works from the browser)
Request everything in **EPSG:3857**. `bbox = [minX, minY, maxX, maxY]` in web-mercator metres.

### US: USGS 3DEP ImageServer
```
https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage
  ?f=image&format=tiff&pixelType=F32&bboxSR=3857&imageSR=3857
  &bbox={minX},{minY},{maxX},{maxY}&size={W},{H}
  &interpolation=RSP_BilinearInterpolation
```
(`format=lerc` is smaller and faster if you add a LERC decoder. TIFF works with geotiff.js.)

### Canada: NRCan HRDEM mosaic (1–2 m lidar DTM), WCS 1.1.1
```
https://datacube.services.geo.ca/wrapper/ogc/elevation-hrdem-mosaic
  ?service=WCS&version=1.1.1&request=GetCoverage&identifier=dtm&format=image/geotiff
  &BoundingBox={minX},{minY},{maxX},{maxY},urn:ogc:def:crs:EPSG::3857
  &GridBaseCRS=urn:ogc:def:crs:EPSG::3857&GridOffsets={step},-{step}
```
The returned size may not be exactly W×H. Resample to the grid you want.

### Merge rule
Fetch both in parallel with `Promise.allSettled`. Per cell, **prefer the HRDEM value when present, else use 3DEP**. (3DEP returns coarse filler in Canada.) Treat values outside -500..9000 as NaN (no data). If fewer than 20% of cells are valid, show "No lidar here."

### Grid sizing
- Cap the view at ~3.2 km across (ask the user to zoom in otherwise). Use at most ~700 cells per side.
- Ground cell size = mercator cell × cos(latitude). Aim for ≥1 m ground cells. Use **ground** metres for areas, distances and the relative-elevation radius.

## Analysis (run in a Web Worker if the project has a build step; ~0.5M cells)
1. **Depression fill:** priority-flood (Barnes et al. 2014) with a binary min-heap. Seed every edge cell and NaN cell. For each popped cell `p` and unvisited neighbour `q`: `fill[q] = max(z[q], fill[p] + 1e-4)`. The epsilon lets flats drain.
2. **Pooling depth (frost pockets):** `depth = max(0, fill - z - 0.005)`. Show only ≥ 0.1 m.
3. **Cold-air drainage:** D8 flow accumulation on the filled surface. Sort cells by `fill` descending, pass each cell's accumulation to its steepest-descent neighbour (divide diagonal drops by √2). Upslope area = acc × ground cell area.
4. **Relative elevation:** `z − mean(z within ~75 m)` using an integral image (summed-area table). Ignore NaN. Negative = lower and colder, positive = higher and warmer.

## Layers (canvas → image overlay aligned to the 3857 bbox)
| Layer | Rendering |
|---|---|
| Frost pockets | depth 0.1 → 3 m+, log ramp `#b3e5ff → #2b7bff → #3b00a8`, more opaque as deeper |
| Drainage paths | cells with upslope area ≥ threshold. Slider 10^2–10^5 m² (default ~2000 m²). Light cyan → bright cyan as area grows |
| Relative elevation (off by default) | diverging blue–white–red, ±4 m |

Toggle each layer, plus an opacity slider. Add a short legend for each. Satellite imagery basemap.

## Interactions
- **"Analyze this view"** button with progress/status text. Afterwards show the data source, cell size, and the **hectares of frost pocket in view**.
- **Click a point:** elevation, pool depth (label it "frost pocket" if ≥ 0.1 m), relative elevation with a colder/warmer note, upslope drainage area (m² or ha).
- **Profile tool:** click start and end, sample ~300 points. Chart the ground as a filled brown area and the filled (cold-air) surface as a blue line, with blue shading where cold air ponds. Caption: "A pond against a rise downhill = a cold-air dam."
- If the project already has vineyard block boundaries: report the **% of each block in a frost pocket** and the max pool depth. Highlight blocks that receive large drainage areas.

## Caveats to show in the UI
- Terrain-only model of radiation frost on calm, clear nights. Not a temperature forecast. Advective (windy) freezes behave differently.
- Bare-earth lidar removes trees, fences and buildings, so it **misses vegetation and fence cold-air dams**. Verify in the field.
- Ponds and reservoirs look like deep pockets (flat water surface). Optionally mask flat areas larger than ~0.5 ha with near-zero relief.

## Done when
- [ ] Niagara-on-the-Lake area, ON (≈ 43.155, -79.07, ~1 km view): loads from **NRCan HRDEM** in a few seconds at ~2 m cells, and shows cold air ponding against the highway embankment.
- [ ] Napa area, CA (≈ 38.39, -122.33): loads from **USGS 3DEP**. Drainage lines follow creeks. The profile chart draws correctly.
- [ ] A view that's too large shows a "zoom in" message. A spot with no lidar shows a clear message. No uncaught errors in the console.
- [ ] Existing project features still work. Run the project's existing tests/lint, if any.
- [ ] Attribution on the map: "Elevation: USGS 3DEP, NRCan HRDEM (Open Government Licence – Canada)".
