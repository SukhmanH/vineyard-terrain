# Deploying the analyzer

The app splits into two pieces that deploy separately:

- **The page** is static (one HTML file, Leaflet from a CDN). It ships inside
  the vineyard site repo at `public/frost/` and is served with the rest of the
  site at `hbbrosvineyards.com/frost`.
- **The API** is this FastAPI app. It needs a container with a real disk.

## Why the API cannot be serverless

Measured from a working virtualenv:

| Package                   | Unpacked |
| ------------------------- | -------- |
| scipy (with scipy.libs)   | 134 MB   |
| rasterio (with GDAL libs) | 93 MB    |
| numpy                     | 34 MB    |
| matplotlib                | 32 MB    |
| whitebox                  | 198 MB   |

That is roughly 490 MB before the app itself. A Vercel or Lambda function caps
at 250 MB unpacked, so the dependency set alone rules it out. Three more things
would break even if it fit:

1. Serverless request bodies cap around 4.5 MB. LidarBC tiles run 4 to 10 MB,
   and a merged block is several of them at once.
2. The design writes `output/<job_id>/` and serves those layers back on later
   requests. Serverless filesystems do not persist between invocations.
3. A single tile takes seconds, a multi-tile mosaic takes minutes. Short
   function timeouts cut that off.

A small always-on container is the right shape, and it is cheap.

## Fly.io

`fly.toml` is committed. From the repo root:

```sh
fly launch --no-deploy --name frost-api
fly volumes create vta_data --size 10 --region sea
fly secrets set VTA_ALLOWED_ORIGINS="https://hbbrosvineyards.com"
fly deploy
```

Then point the page at it and rebuild the site:

```sh
FROST_API_BASE=https://frost-api.fly.dev npm run build
```

## Any other container host

The `Dockerfile` has no Fly-specific parts, so Render, Railway, Cloud Run or a
plain VPS all work:

```sh
docker build -t frost-api .
docker run -p 8000:8000 -v frost-data:/data \
  -e VTA_ALLOWED_ORIGINS="https://hbbrosvineyards.com" frost-api
```

## Beside Hermes on `vineyard` (no container)

The parcel API for Hermes runs on the Hermes host itself as a systemd **user**
service next to `hermes-gateway`, reachable on the tailnet only
(`100.71.243.94:8000`). That box is a Core 2 Duo E8400 with no SSE4.2, so
PyPI's numpy and scipy wheels, built for newer CPUs, can die there with
"Illegal instruction". The scientific stack comes from Ubuntu's packages (built
for the generic x86-64 baseline) and pip adds only `whitebox`, held to those
versions by `deploy/constraints-core2.txt`.

```sh
sudo apt install python3-venv python3-numpy python3-scipy python3-rasterio \
  python3-matplotlib python3-shapely python3-fastapi python3-uvicorn python3-multipart
git clone -b parcel-frost vineyard-terrain.bundle ~/vineyard-terrain   # or a git remote
cd ~/vineyard-terrain
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -c deploy/constraints-core2.txt whitebox pytest
.venv/bin/python -c "import whitebox; whitebox.WhiteboxTools().version()"  # fetches the binary once
.venv/bin/python -m pytest -q        # must pass, Whitebox included, before the service starts

install -d -m 700 ~/.config/frost-api ~/frost-data
cat > ~/.config/frost-api/env <<EOF   # chmod 600; the token also goes in Hermes' FROST_API_TOKEN
VTA_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
VTA_DATA_DIR=$HOME/frost-data
VTA_MAP_BASE_URL=http://100.71.243.94:8000
VTA_TILE_CACHE_MB=10000
VTA_WBT_MAX_PROCS=1
EOF
chmod 600 ~/.config/frost-api/env
VTA_DATA_DIR=~/frost-data .venv/bin/python -m backend.pipeline.tiles refresh --sheet 082e
cp deploy/frost-api.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now frost-api
curl -s http://100.71.243.94:8000/api/health
```

The unit runs at `Nice=19`, `CPUWeight=10`, an idle I/O class and
`MemoryMax=3G`, so the gateway and local transcription always come first. It
binds the tailnet address, which exists only once `tailscaled` is up; until
then the bind fails and systemd retries every 10 s. Logs:
`journalctl --user -u frost-api -f`.

WhiteboxTools runs on one thread there (`VTA_WBT_MAX_PROCS=1`): on this
2-core box v2.4.0's FillDepressions panicked about 1 run in 30 on two threads
and never on one, and the host's test suite failed a different routing test on
every run until it was capped.
To update: copy a new bundle, `git pull` in `~/vineyard-terrain`, then
`systemctl --user restart frost-api` (unfinished jobs are re-queued).

## Configuration

| Variable               | Default        | Purpose                                                                                   |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------- |
| `VTA_DATA_DIR`         | the repo root  | Where `uploads/`, `output/` and `tiles/` live. Point at a mounted volume so jobs and the tile cache survive a restart. |
| `VTA_ALLOWED_ORIGINS`  | unset          | Comma-separated CORS allowlist. Unset means same-origin only, which is what local dev wants. |
| `VTA_API_TOKEN`        | unset          | Bearer token for `POST /api/parcel-jobs`. Unset means parcel jobs are disabled (503), never open. |
| `VTA_MAP_BASE_URL`     | the request's base URL | Base of the `map_url` a parcel job returns, e.g. `http://100.71.243.94:8000`. |
| `VTA_SHEETS`           | `082e`         | Comma-separated BCGS sheets in the tile index. |
| `VTA_TILE_CACHE_MB`    | `4000`         | Size cap of the downloaded tile cache; least recently used tiles are dropped first. |
| `VTA_PARCEL_BUFFER_M`  | `1000`         | Default AOI buffer around a parcel (a request's `buffer_m` overrides it). |
| `VTA_GIT_SHA`          | `git rev-parse` | Commit recorded in each parcel job's provenance; set it where there is no `.git`. |
| `VTA_WBT_MAX_PROCS`    | Whitebox's own | Caps WhiteboxTools' threads, re-applied every job (Whitebox stores it in its `settings.json`). `1` on `vineyard`. |

`GET /api/health` returns `{"ok": true, "jobs": N, "frost_model": "v2",
"model_version": "v2+z1", "parcel_jobs": true}`. The container healthcheck
uses it, and so does the page: when the API does not answer, the upload box is
replaced with an "analysis service offline" notice and the map still works.

## Parcel jobs

`POST /api/parcel-jobs` (bearer token) takes `{address?, town?, pid?, lat?, lon?,
geometry?, listing_acres?, buffer_m?, ref?, confirmed?, clss_id?}` and answers 202
`{job_id, status, map_url}`. `GET /api/parcel-jobs/{job_id}` needs no token (the
32-character random id is the capability) and returns the status, the parcel, its
metrics, provenance and the map link. The page opens a job with `/?job=<id>`.

- **Run one uvicorn worker process.** Jobs run one at a time on a worker thread
  inside the app; a second process would start a second queue.
- Status lives in `output/<id>/status.json`; a restart re-queues unfinished jobs, and a
  job interrupted twice is failed rather than retried forever.
- Tiles come from the LidarBC object store into `tiles/cache/` (capped by
  `VTA_TILE_CACHE_MB`). Build the index once, and again after a new LidarBC delivery:

      python -m backend.pipeline.tiles refresh --sheet 082e    # ~2 minutes, ~6,000 headers

- Parcel PNG layers are pruned after 30 days; the JSON is kept.
- The only outbound hosts are the object store, the DataBC WFS and the BC geocoder
  (`backend/pipeline/net.py`).

## Sizing the volume

Job folders are large. Each keeps eight rendered PNG layers, which measured
28 MB for a single tile and 90 MB for a four-tile mosaic. Hydrology scratch
files are larger still but are deleted when the job ends, including when it
fails. A 10 GB volume holds roughly a hundred single-tile jobs. Clearing a
block in the UI issues `DELETE /api/jobs/<id>`, which removes both the job
folder and its uploaded tiles.

## Cost note

The pipeline is CPU-bound and memory-hungry: it holds several float32 copies of
a 1900 x 1480 grid at once, and a mosaic multiplies that. Two shared CPUs and
2 GB of RAM handle single tiles comfortably. Give it 4 GB before merging many
tiles in one block.
