"""Parcel scoring jobs: a queue, one worker thread, a status file per job.

POST /api/parcel-jobs only enqueues. One worker runs jobs one at a time, because the API
host has 2 to 3 GB for the pipeline and a parcel AOI is the size of the four-tile mosaic it
already handles. Each job's progress lives in output/<id>/status.json, so it survives a
restart and needs no database:

    queued -> resolving -> needs_confirmation
                        -> fetching -> analyzing -> done
    (any) -> failed, with the error

Job ids are 32 random hex characters: a map link works without the API token but cannot be
guessed. A job interrupted twice (the service killed mid-run, say by its memory cap) is
failed rather than retried forever.
"""
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

from .pipeline import net, run, tiles
from .pipeline import parcel as parcels

log = logging.getLogger("vta.jobs")

STATUSES = ("queued", "resolving", "needs_confirmation", "fetching", "analyzing", "done",
            "failed")
TERMINAL = ("needs_confirmation", "done", "failed")
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
QUEUE_MAX = 50
MAX_ATTEMPTS = 2
LAYER_RETENTION_DAYS = 30


class QueueFull(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _retry(fn, tries=25):
    """Windows refuses to replace (or open) a file another thread has open for a moment,
    as when a status poll reads status.json while the worker rewrites it."""
    for i in range(tries):
        try:
            return fn()
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.01 * (i + 1))


class ParcelJobs:
    def __init__(self, output_dir, resolver=None, runner=None):
        self.output = output_dir
        self.resolver = resolver or parcels.resolve
        self.runner = runner or run.run_parcel
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.worker = None

    # ----------------------------------------------------------- status files
    def job_dir(self, job_id):
        if not JOB_ID_RE.fullmatch(job_id or ""):
            raise ValueError("bad parcel job id")
        return os.path.join(self.output, job_id)

    def read_status(self, job_id):
        path = os.path.join(self.job_dir(job_id), "status.json")

        def read():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        try:
            return _retry(read)
        except (OSError, ValueError):
            return None

    def write_status(self, job_id, **changes):
        with self.lock:
            st = self.read_status(job_id) or {"job_id": job_id, "created_at": _now()}
            st.update(changes, updated_at=_now())
            path = os.path.join(self.job_dir(job_id), "status.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(st, f, indent=1)
            _retry(lambda: os.replace(tmp, path))
            return st

    def _fail(self, job_id, error):
        log.warning("parcel job %s failed: %s", job_id, error)
        self.write_status(job_id, status="failed", error=str(error))

    # ------------------------------------------------------------- the queue
    def pending(self):
        n = 0
        for st in self._all_status():
            if st.get("status") not in TERMINAL:
                n += 1
        return n

    def submit(self, request):
        """Queue a parcel request (already validated by the API). Returns the job id."""
        if self.pending() >= QUEUE_MAX:
            raise QueueFull(f"{QUEUE_MAX} parcel jobs are already waiting")
        job_id = secrets.token_hex(16)
        self.write_status(job_id, status="queued", request=request,
                          ref=request.get("ref"), attempts=0, error=None)
        self.queue.put(job_id)
        return job_id

    def start(self):
        """Recover jobs a restart interrupted, then start the worker (once)."""
        with self.lock:
            if self.worker is not None:
                return
            self.worker = threading.Thread(target=self._work, name="parcel-jobs",
                                           daemon=True)
        for st in sorted(self._all_status(), key=lambda s: s.get("created_at", "")):
            if st.get("status") in TERMINAL:
                continue
            if st.get("attempts", 0) >= MAX_ATTEMPTS:
                self._fail(st["job_id"], "interrupted twice (the service restarted "
                                         "mid-run); not retried")
            else:
                self.write_status(st["job_id"], status="queued")
                self.queue.put(st["job_id"])
        self.prune()
        self.worker.start()

    def _work(self):
        while True:
            job_id = self.queue.get()
            try:
                self.run_job(job_id)
            except Exception:                 # never let one job kill the worker
                log.error("parcel job %s crashed:\n%s", job_id, traceback.format_exc())
                self._fail(job_id, "internal error (see server log)")
            finally:
                self.queue.task_done()
                try:
                    self.prune()
                except OSError:
                    pass

    # --------------------------------------------------------------- a job
    def run_job(self, job_id):
        st = self.read_status(job_id)
        if st is None or st.get("status") in TERMINAL:
            return
        req = st.get("request") or {}
        self.write_status(job_id, status="resolving", attempts=st.get("attempts", 0) + 1)
        try:
            res = self.resolver(req)
        except parcels.BoundaryError as exc:
            return self._fail(job_id, f"bad boundary: {exc}")
        except ValueError as exc:
            return self._fail(job_id, str(exc))
        except (net.NetError, parcels.ResolveError) as exc:
            return self._fail(job_id, f"parcel lookup failed: {exc}")

        out_dir = self.job_dir(job_id)
        public = parcels.public(res)
        if res["status"] != "resolved":
            with open(os.path.join(out_dir, "parcel.geojson"), "w", encoding="utf-8") as f:
                json.dump(parcels.feature_collection(res), f)
            self.write_status(job_id, status="needs_confirmation", parcel=public)
            return

        self.write_status(job_id, status="fetching", parcel=public)
        label = req.get("address") or res.get("pid") or "parcel"
        try:
            stats, provenance = self.runner(
                res, out_dir, buffer_m=req.get("buffer_m"), label=label,
                on_status=lambda s: self.write_status(job_id, status=s))
        except (run.ParcelAnalysisError, tiles.TileError, net.NetError) as exc:
            return self._fail(job_id, str(exc))
        self.write_status(job_id, status="done", metrics=stats["parcel"],
                          provenance=provenance, error=None)

    # ------------------------------------------------------------- reading
    def _all_status(self):
        if not os.path.isdir(self.output):
            return []
        out = []
        for name in os.listdir(self.output):
            if JOB_ID_RE.fullmatch(name):
                st = self.read_status(name)
                if st is not None:
                    out.append(st)
        return out

    def view(self, job_id, map_base):
        """What GET /api/parcel-jobs/{id} returns."""
        st = self.read_status(job_id)
        if st is None:
            return None
        return {
            "job_id": job_id,
            "status": st.get("status"),
            "error": st.get("error"),
            "ref": st.get("ref"),
            "created_at": st.get("created_at"),
            "updated_at": st.get("updated_at"),
            "parcel": st.get("parcel"),
            "provenance": st.get("provenance"),
            "metrics": st.get("metrics"),
            "layers_pruned": bool(st.get("layers_pruned")),
            "map_url": f"{map_base.rstrip('/')}/?job={job_id}",
        }

    def prune(self, now=None):
        """Drop the PNG layers of parcel jobs finished over LAYER_RETENTION_DAYS ago. The
        JSON (status, stats, meta, bounds, parcel.geojson) is kept."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=LAYER_RETENTION_DAYS)
        pruned = []
        for st in self._all_status():
            if st.get("status") not in ("done", "failed") or st.get("layers_pruned"):
                continue
            if _parse(st.get("updated_at") or _now()) > cutoff:
                continue
            layers = os.path.join(self.job_dir(st["job_id"]), "layers")
            if os.path.isdir(layers):
                for name in os.listdir(layers):
                    if name.endswith(".png"):
                        os.remove(os.path.join(layers, name))
            with self.lock:
                path = os.path.join(self.job_dir(st["job_id"]), "status.json")
                st["layers_pruned"] = True
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(st, f, indent=1)
                _retry(lambda: os.replace(tmp, path))
            pruned.append(st["job_id"])
        return pruned
