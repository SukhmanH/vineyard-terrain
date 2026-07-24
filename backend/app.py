"""FastAPI app. Run: uvicorn backend.app:app --reload --port 8000"""
import json
import os
import re
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import run as pipeline_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = os.path.join(ROOT, "uploads")
OUTPUT = os.path.join(ROOT, "output")
FRONTEND = os.path.join(ROOT, "frontend")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

app = FastAPI(title="Vineyard Terrain Analyzer")


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
