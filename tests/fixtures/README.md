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
