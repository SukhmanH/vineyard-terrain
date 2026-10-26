# Data sources for the parcel path

Every source below was checked with a live call before code was built on it (October 2026).
Field names in the code are the ones seen in these responses; nothing is guessed. Trimmed
copies of the responses used by the tests live in `tests/fixtures/`.

The server may only reach the three HTTPS hosts in `backend/pipeline/net.py`
(`ALLOWED_HOSTS`): the object store, the DataBC WFS and the BC Address Geocoder. NRCan is
reached once, by hand, by the CLSS import (`backend/pipeline/clss.py`).

## LidarBC object store

`https://nrs.objectstore.gov.bc.ca/gdwuts`, public, S3 ListObjects **v1** XML.

- Keys: `<block>/<sheet>/<year>/<product>/<file>`, e.g.
  `082/082e/2024/dem/bc_082e023_1_2_1_xli1m_utm11_20240721_20240726.tif`.
- Products under a year: `dem`, `dsm`, `pointcloud`, `orthophoto`. Only `dem` (bare earth)
  is ever analyzed. `dsm` is fetched only as a DEM's twin for the informational canopy
  height. **DEM file names carry no `_dtm` suffix**, so the product comes from the path.
- Folder markers are objects too: `082/082e/2024/dem/_$folder$` (size 0). They are skipped.
- Paging: `IsTruncated` plus `Marker`. Without a `delimiter`, no `NextMarker` comes back,
  so the last key of a page is the next marker. 1,000 keys per page.
- `?prefix=082/082e/&delimiter=/` lists the delivery years as `CommonPrefixes`.

```xml
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>gdwuts</Name>
<Prefix>082/082e/2024/dem/</Prefix><Marker></Marker><MaxKeys>3</MaxKeys>
<IsTruncated>true</IsTruncated>
<Contents><Key>082/082e/2024/dem/bc_082e002_1_1_2_xli1m_utm11_20240721_20250719.tif</Key>
<LastModified>2026-05-07T23:28:22.627Z</LastModified>
<ETag>"8f44b012193b24a366305f463324d9d8"</ETag><Size>8786803</Size>...</Contents>
```

ETags of small files are plain MD5s, so whole-tile downloads are verified against them.
Multipart uploads have `md5-N` ETags (the 2018 mapsheets) and are only size-checked.

### What 082e holds (October 2026)

| Year | DEM files | Size | Naming and shape | CRS |
|---|---|---|---|---|
| 2017 | 351 | 1.5 GB | 1:2,500 tiles, `bc_082e003_1_4_4_xl1m_17603.tif`, `bc_082e053_1_3_2_xl1m_utm11_170606.tif`; 512 px tiles, deflate, overviews | EPSG:26911 (NAD83 / UTM 11N, no vertical CRS) |
| 2018 | 60 (47 names) | 10.3 GB | sheet-only names, `bc_082e003_xli1m_utm11_2018.tif`; 13 names also have a different `... (2).tif` (e.g. 96 MB vs 253 MB, different footprints); sizes range from one tile to a whole mapsheet | EPSG:2955 |
| 2019 | 41 | 20.6 GB | whole 1:20,000 mapsheets, `bc_082e002_xli1m_utm11_2019.tif` 15108 x 11737, striped, uncompressed (~709 MB) | EPSG:2955 |
| 2022 | 1 | 0.3 GB | whole mapsheet `bc_082e095_xli1m_utm11_2022.tif`, DSM twin | |
| 2024 | 2,056 | 15.9 GB | 1:2,500 tiles, `..._xli1m_utm11_20240721_20240726.tif`; some reprocessed as `..._20240721_20250719.tif`; DSM twins `..._dsm.tif` | EPSG:6654 |
| 2025 | 3,454 | 27.3 GB | 1:2,500 tiles, `..._20250726_20250726.tif`, DSM twins | EPSG:6654 |

- EPSG:6654 is **NAD83(CSRS) / UTM 11N + CGVD2013 height**: the 2024/25 heights are on
  CGVD2013, and the 2017 files declare no vertical datum. Expect real elevation steps where
  2017 meets 2024/25, which is what the seam check (`SEAM_MAX_STEP_M = 0.15`) catches.
- PROJ as bundled here applies no shift between NAD83 (26911) and NAD83(CSRS) (2955, 6654):
  both transform to identical BC Albers coordinates.
- 2024/25 tiles: about 1910 x 1489 at 1 m, striped one row per strip, LZW, no overviews,
  ~8 MB. Valid data is clipped to the BCGS cell (~252 ha of the ~284 ha bbox), so adjacent
  tiles abut and do not overlap.
- Inside 2017, 2024 and 2025 each tile stem occurs once. Across years the same footprint
  repeats (2017 and 2024 often both cover a cell).
- Headers read over GDAL `/vsicurl/` take 0.1 to 0.7 s, so the index reads every DEM header
  (the footprint, CRS and size come from the file, never the name).

The six properties sit on: Upper Bench 082E053112 (2017, 2024), Naramata 082E053312 (2017,
2024), Rust 082E013113 (2017), Tucelnuit 082E023121 (2017, 2024), Hwy 97 082E013112 (2017),
Cassini 082E013121 (2017). 2025 covers none of them.

## DataBC WFS

`https://openmaps.gov.bc.ca/geo/pub/wfs`, WFS 2.0.0, `outputFormat=application/json`,
`srsName=EPSG:3005`.

**Paging:** `startIndex` without `sortBy` is HTTP 400, and `sortBy` (OBJECTID or
PMBC_PP_SYSID) on the parcel layer runs into a 504 gateway timeout at 60 s. Unsorted spatial
queries answer in ~0.3 s. So parcel queries are one bounded page (`count`), and a response
whose `numberMatched` is over `count` is an error rather than a silently short list.

### BCGS 1:2,500 grid

`pub:WHSE_BASEMAPPING.BCGS_2500_GRID`. DescribeFeatureType: `MAP_TILE`,
`MAP_TILE_DISPLAY_NAME`, `FCODE`, `GEOMETRY`, `OBJECTID`, `FEATURE_AREA_SQM`,
`FEATURE_LENGTH_M`. Tile `bc_082e053_1_3_2` is `MAP_TILE = "082E053132"`,
`MAP_TILE_DISPLAY_NAME = "82E.053.1.3.2"`, about 251.6 ha. Used only as a cross-check:
footprints come from the GeoTIFF headers because the 2018/2019/2022 names do not encode
1:2,500 tiles.

### ParcelMap BC parcels

`pub:WHSE_CADASTRE.PMBC_PARCEL_POLY_SV`. DescribeFeatureType: `PMBC_PP_SYSID`,
`PARCEL_NAME`, `PLAN_NUMBER`, `PIN`, `PID`, `PID_FORMATTED`, `PID_NUMBER`, `PARCEL_STATUS`,
`PARCEL_CLASS`, `OWNER_TYPE`, `PARCEL_START_DATE`, `MUNICIPALITY`, `REGIONAL_DISTRICT`,
`WHEN_UPDATED`, `FEATURE_AREA_SQM`, `FEATURE_LENGTH_M`, `SHAPE` (geometry), `OBJECTID`.

Values seen in a 2,000-parcel south Okanagan sample:

- `OWNER_TYPE`: Private 1587, Unclassified 290, Local Government 84, Crown Provincial 27,
  Untitled Provincial 9, Crown Agency 2, Federal 1.
- `PARCEL_CLASS`: Subdivision 1651, Interest 245, Road 57, Building Strata 36, Park 5,
  Return To Crown 3, Crown Subdivision 1, Part of Primary 1, Primary 1.

```json
{"type":"Feature","geometry":null,"properties":{"PMBC_PP_SYSID":4915276,
 "PARCEL_NAME":"012525499","PLAN_NUMBER":"KAP307","PID_FORMATTED":"012-525-499",
 "PARCEL_STATUS":"Active","PARCEL_CLASS":"Subdivision","OWNER_TYPE":"Private",
 "MUNICIPALITY":"Penticton, The Corporation of the City of","FEATURE_AREA_SQM":36451.3914}}
```

**Reserve land:** a point on the Black Sage bench returns `PARCEL_NAME "PART: See
PIN13431860"`, `PLAN_NUMBER "4TRIR_IR_OSOYOOS"`, `PARCEL_CLASS "Part of Primary"`,
`OWNER_TYPE "Federal"`, 32,069 ac: Osoyoos IR 1 is one polygon with no internal lots or
leases, as expected. A point west of the river channel at Penticton returns another single
Federal "Part of Primary" polygon (36,126 ac, plan `11TRIR_IR_OSOYOOS`): the "OSOYOOS" in
plan numbers is the Osoyoos Division land district, not the reserve.

## BC Address Geocoder

`https://geocoder.api.gov.bc.ca/addresses.json`, no key needed. Parameters used:
`addressString`, `maxResults`, `outputSRS=3005` (integer metres), `locationDescriptor=parcelPoint`.

Fields read from `features[].properties`: `fullAddress`, `score`, `matchPrecision`,
`precisionPoints`, `faults` (`[{value, element, fault, penalty}]`), `localityName`,
`locationDescriptor`, `locationPositionalAccuracy`.

```json
{"fullAddress":"1056 Naramata Rd, Penticton, BC","score":100,"matchPrecision":"CIVIC_NUMBER",
 "precisionPoints":100,"faults":[],"localityName":"Penticton","locationDescriptor":"parcelPoint",
 "locationPositionalAccuracy":"high"}
```

Only `CIVIC_NUMBER` matches with `score >= 90` and no LOCALITY fault are accepted. The case
that proves score alone is not enough: `4798 Okanagan Hwy, Oliver, BC` (Cassini) comes back
as `4798 Hwy 97, Peachland, BC`, score 87, `BLOCK`, faults `LOCALITY:missing`,
`STREET_NAME:isHighwayAlias`, ~70 km from the property. `70 Upper Bench Rd` matches
`70 Upper Bench Rd S` at 98 with `STREET_DIRECTION:missing`, which is fine.

The geocoder's parcels resource is restricted to BC ministries, so parcels come from the WFS.

## NRCan Canada Lands (CLSS)

Reserve lots are not in ParcelMap BC. NRCan publishes per-reserve cadastral shapefiles on
`ftp://ftp.nrcan.gc.ca/ess/sgb_pub/sgb_datasets/bc/`. The folder names are not the reserve
short names: `OSOYOOS_1` does not exist.

- `OSOYOOS_INDIAN_RESERVE_NO_1/OSOYOOS_INDIAN_RESERVE_NO_1_SHP.zip`, 2,060,332 bytes, Sep 21
- `PENTICTON_1/PENTICTON_1_SHP.zip`, 3,113,356 bytes, Oct 12

Also present: `OSOYOOS_3`, `PENTICTON_2`, `PENTICTON_3A`, and DWG and French/English copies.
Imported once into `data/clss/` with the Open Government Licence - Canada notice (see
`data/clss/README.md` and `backend/pipeline/clss.py`).

## Not used

The community STAC collection `stac-elevation-bc` at `https://images.a11s.one` was not needed:
the object store listing plus headers is the source of truth.
