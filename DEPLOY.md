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

## Configuration

| Variable               | Default        | Purpose                                                                                   |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------- |
| `VTA_DATA_DIR`         | the repo root  | Where `uploads/` and `output/` live. Point at a mounted volume so jobs survive a restart.   |
| `VTA_ALLOWED_ORIGINS`  | unset          | Comma-separated CORS allowlist. Unset means same-origin only, which is what local dev wants. |

`GET /api/health` returns `{"ok": true, "jobs": N}`. The container healthcheck
uses it, and so does the page: when the API does not answer, the upload box is
replaced with an "analysis service offline" notice and the map still works.

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
