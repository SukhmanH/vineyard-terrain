# Canada Lands Survey System (CLSS) reserve lots

ParcelMap BC carries each Indian reserve as a single Federal polygon, so lots inside a reserve
(band leasehold, OIB and PIB leases) are not in it. These are NRCan's per-reserve cadastral
datasets, downloaded once with `python -m backend.pipeline.clss fetch` and read straight from
the zips by `backend/pipeline/clss.py` (layer `Land_Parcel`, lot id `PIN`).

| Folder | Reserve | Lots | Dataset date |
|---|---|---|---|
| `OSOYOOS_INDIAN_RESERVE_NO_1/` | Osoyoos Indian Reserve No. 1 | 925 (32,158 ac) | 2026-09-21 |
| `PENTICTON_1/` | Penticton Indian Reserve No. 1 | 1,220 (47,898 ac) | 2026-10-12 |

Each folder's `SOURCE.md` records the exact URL, size and retrieval date.

## Licence

Contains information licensed under the Open Government Licence - Canada.
https://open.canada.ca/en/open-government-licence-canada

Source: Natural Resources Canada, Surveyor General Branch, Canada Lands Survey System
cadastral datasets (`ftp://ftp.nrcan.gc.ca/ess/sgb_pub/sgb_datasets/bc/`).

A lot found here is offered to the owner as a candidate only. It is never scored without
confirmation, and survey fabric is not proof of a lease or of its terms.
