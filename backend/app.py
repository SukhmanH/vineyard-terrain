"""FastAPI app. Run: uvicorn backend.app:app --reload --port 8000"""
import hmac
import json
import os
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import jobs
from .pipeline import run as pipeline_run
from .pipeline import scoring, zonal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VTA_DATA_DIR points uploads/ and output/ at a mounted volume when the app
# runs in a container. Falls back to the repo folders for local development.
DATA = os.environ.get("VTA_DATA_DIR") or ROOT
UPLOADS = os.path.join(DATA, "uploads")
OUTPUT = os.path.join(DATA, "output")
FRONTEND = os.path.join(ROOT, "frontend")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# Parcel scoring jobs run one at a time on a worker thread (backend/jobs.py). Run the API
# with ONE uvicorn worker process: a second process would start a second queue.
PARCEL_JOBS = jobs.ParcelJobs(OUTPUT)


@asynccontextmanager
async def lifespan(_app):
    PARCEL_JOBS.start()
    yield


app = FastAPI(title="Vineyard Terrain Analyzer", lifespan=lifespan)

# The static frontend is served from the vineyard site (hbbrosvineyards.com
# /frost) while this API runs on its own host, so browsers send cross-origin
# requests. VTA_ALLOWED_ORIGINS is a comma-separated allowlist; unset means
# same-origin only, which is what local development wants.
_origins = [o.strip() for o in
            os.environ.get("VTA_ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/api/health")
def health():
    """Liveness probe for the container host, and the check the /frost page
    uses to tell the visitor whether the analysis service is reachable.
    model_version tells a client (Hermes) which frost model its scores come from."""
    return {"ok": True, "jobs": len(os.listdir(OUTPUT)), "frost_model": scoring.MODEL,
            "model_version": zonal.MODEL_VERSION,
            "parcel_jobs": bool(os.environ.get("VTA_API_TOKEN"))}


# ------------------------------------------------------------------ parcel jobs
class ParcelJobRequest(BaseModel):
    """A listing to score. Structured fields only: free text from a listing email is
    never sent here, and nothing in a request is ever fetched as a URL."""
    model_config = ConfigDict(extra="forbid")

    address: str | None = Field(None, max_length=200)
    town: str | None = Field(None, max_length=80)
    pid: str | list[str] | None = None
    lat: float | None = Field(None, ge=48.0, le=51.0)
    lon: float | None = Field(None, ge=-121.0, le=-118.0)
    geometry: dict | str | None = None
    listing_acres: float | None = Field(None, gt=0, le=5000)
    buffer_m: float | None = Field(None, ge=200, le=2000)
    ref: str | None = Field(None, max_length=120)
    confirmed: bool = False
    clss_id: str | None = Field(None, max_length=80)

    @model_validator(mode="after")
    def _check(self):
        if not (self.geometry or self.clss_id or self.pid or self.address
                or (self.lat is not None and self.lon is not None)):
            raise ValueError("a parcel needs a geometry, PID, lat/lon or address")
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon go together")
        pids = [self.pid] if isinstance(self.pid, str) else (self.pid or [])
        if len(pids) > 10 or any(len(p) > 40 for p in pids):
            raise ValueError("at most 10 PIDs")
        if self.geometry is not None and len(json.dumps(self.geometry)) > 1_000_000:
            raise ValueError("geometry is over 1 MB")
        return self


def _require_token(authorization):
    """Parcel jobs make the server download data, so they need VTA_API_TOKEN."""
    token = os.environ.get("VTA_API_TOKEN")
    if not token:
        raise HTTPException(503, "parcel jobs are disabled: VTA_API_TOKEN is not set")
    scheme, _, value = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise HTTPException(401, "bearer token required",
                            headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(value.strip().encode(), token.encode()):
        raise HTTPException(403, "invalid token")


def _map_base(request):
    return os.environ.get("VTA_MAP_BASE_URL") or str(request.base_url)


@app.post("/api/parcel-jobs", status_code=202)
def create_parcel_job(body: ParcelJobRequest, request: Request,
                      authorization: str | None = Header(None)):
    _require_token(authorization)
    try:
        job_id = PARCEL_JOBS.submit(body.model_dump(exclude_none=True))
    except jobs.QueueFull as exc:
        raise HTTPException(429, str(exc))
    return {"job_id": job_id, "status": "queued",
            "map_url": f"{_map_base(request).rstrip('/')}/?job={job_id}"}


@app.get("/api/parcel-jobs/{job_id}")
def get_parcel_job(job_id: str, request: Request):
    """No token: the 32-character random id is the capability, so the map link works."""
    if not jobs.JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(404, "no such parcel job")
    view = PARCEL_JOBS.view(job_id, _map_base(request))
    if view is None:
        raise HTTPException(404, "no such parcel job")
    return view


@app.post("/api/upload")
async def upload(file: list[UploadFile] = File(...),
                 boundary: str = Form(None)):
    """Upload one tile or several tiles that make up one block. Multiple tiles
    are reprojected to a common CRS and merged before analysis. An optional
    GeoJSON polygon in the "boundary" field clips the analysis to a parcel."""
    files = file if isinstance(file, list) else [file]
    for f in files:
        if not f.filename.lower().endswith((".tif", ".tiff")):
            raise HTTPException(400, f"expected GeoTIFFs (.tif); got {f.filename}")

    job_id = uuid.uuid4().hex[:8]
    tif_paths = []
    for i, f in enumerate(files):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename)
        p = os.path.join(UPLOADS, f"{job_id}_{i}_{safe}")
        with open(p, "wb") as out:
            out.write(await f.read())
        tif_paths.append(p)

    geojson = None
    if boundary:
        try:
            geojson = json.loads(boundary)
        except json.JSONDecodeError:
            raise HTTPException(400, "boundary must be valid GeoJSON")

    out_dir = os.path.join(OUTPUT, job_id)
    try:
        # synchronous on purpose: tiles process in seconds
        pipeline_run.run(tif_paths, out_dir, boundary=geojson)
    except Exception as e:  # surface pipeline errors to the client
        raise HTTPException(500, f"pipeline failed: {e}")
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    out_dir = os.path.join(OUTPUT, job_id)
    if not os.path.isdir(out_dir):
        raise HTTPException(404, "no such job")
    def read(name):
        p = os.path.join(out_dir, name)
        with open(p) as f:
            return json.load(f)
    layers_dir = os.path.join(out_dir, "layers")
    layers = sorted(p[:-4] for p in os.listdir(layers_dir)
                    if p.endswith(".png")) if os.path.isdir(layers_dir) else []
    return {
        "meta": read("meta.json"),
        "stats": read("stats.json"),
        "bounds": read("bounds.json"),
        "layers": layers,
    }


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Remove a job's output folder and its uploaded tiles (clear a block)."""
    if not re.fullmatch(r"[0-9a-f]{8}", job_id):
        raise HTTPException(400, "bad job id")
    import shutil
    out_dir = os.path.join(OUTPUT, job_id)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    for name in os.listdir(UPLOADS):
        if name.startswith(job_id + "_"):
            try:
                os.remove(os.path.join(UPLOADS, name))
            except OSError:
                pass
    return {"cleared": job_id}


app.mount("/output", StaticFiles(directory=OUTPUT), name="output")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))
