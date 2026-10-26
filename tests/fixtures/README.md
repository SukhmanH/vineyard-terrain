# Recorded responses

Real responses from the parcel path's data sources, recorded in October 2026 and trimmed so
pytest never touches the network (tests/conftest.py fails any test that tries). See
docs/data-sources.md for what each service returns and why the code reads what it reads.

- `bucket/` LidarBC object store listings (S3 ListObjects v1).
  - `years_082e.xml`: `?prefix=082/082e/&delimiter=/`, as returned.
  - `dem_2024_page1.xml`: `?prefix=082/082e/2024/dem/&max-keys=3`, as returned
    (truncated, folder marker included).
  - `dem_2024_page2.xml`: the next page by marker with `max-keys=2`. **Trimmed:**
    `IsTruncated` set to `false` so the listing ends after two pages.
  - `dem_2018_dupes.xml`: `?prefix=082/082e/2018/dem/bc_082e003`, as returned. The
    `(2)` file is a different raster, not a copy.
- `geocoder/` BC Address Geocoder `addresses.json` with `outputSRS=3005` and
  `locationDescriptor=parcelPoint`. **Trimmed:** only `type`, `queryAddress`,
  `searchTimestamp`, `version`, `crs` and `features` kept (disclaimers dropped).
  - `naramata_1056.json` "1056 Naramata Rd, Penticton, BC": civic, score 100.
  - `cassini_peachland.json` "4798 Okanagan Hwy, Oliver, BC": Peachland, BLOCK, 87.
  - `upper_bench_direction.json` "70 Upper Bench Rd, Penticton, BC": 98, direction missing.
  - `locality_only.json` "Keremeos, BC": a locality-level match.
- `wfs/` DataBC WFS `WHSE_CADASTRE.PMBC_PARCEL_POLY_SV`, `srsName=EPSG:3005`. **Trimmed:**
  coordinates rounded to 0.1 m; `federal_black_sage.json` geometry simplified at 25 m (the
  real Osoyoos IR 1 polygon is far larger); `strata_penticton.json` and
  `road_interest_naramata.json` keep their first 2 features.
  - `at_naramata_1056.json`, `pid_012525499.json`, `bbox_naramata_1056.json` (the parcel and
    its 8 neighbours).
  - `at_cassini.json` (empty: the coordinates are in no parcel) and `near_cassini.json`
    (4 parcels within 60 m).
