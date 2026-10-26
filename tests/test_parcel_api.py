"""Parcel job API (backend/app.py) and job queue (backend/jobs.py).

Handlers are called directly (FastAPI's TestClient would need httpx, a dependency this repo
does not carry). Resolution and analysis are stubbed: they have their own tests.
"""
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from shapely.geometry import box
from starlette.requests import Request

from backend import app as api
from backend import jobs
from backend.pipeline import parcel, run

TOKEN = "s3cret-token"


def _request():
    return Request({"type": "http", "method": "GET", "scheme": "http", "path": "/",
                    "root_path": "", "query_string": b"", "headers": [],
                    "server": ("frost.test", 8000)})


def _resolved(**kw):
    geom = box(1465600, 522600, 1465800, 522800)
    res = {"status": "resolved", "pid": "012-525-499", "pids": ["012-525-499"],
           "geometry_3005": geom, "geometry_wgs84": parcel.to_wgs84(geom),
           "area_acres": 9.88, "owner_type": "Private", "parcel_class": "Subdivision",
           "source": "address", "confidence": "high", "candidates": [], "reasons": [],
           "flags": [], "point_3005": None}
    res.update(kw)
    return res


def _needs_confirmation():
    cand = {"index": 1, "kind": "pmbc", "pid": "011-333-596", "clss_id": None,
            "designator": None, "acres": 9.2, "owner_type": "Private",
            "parcel_class": "Subdivision", "distance_m": 4.6, "acres_off_pct": None,
            "geometry": box(1468200, 478760, 1468400, 478960)}
    return {"status": "needs_confirmation", "pid": None, "pids": [], "geometry_3005": None,
            "geometry_wgs84": None, "area_acres": None, "owner_type": None,
            "parcel_class": None, "source": "point", "confidence": "low",
            "candidates": [cand], "reasons": ["the point is not inside any parcel"],
            "flags": [], "point_3005": (1468203.9, 478760.2)}


def _fake_runner(calls):
    def runner(res, out_dir, buffer_m=None, label=None, on_status=None):
        calls.append({"pid": res["pid"], "buffer_m": buffer_m, "label": label})
        on_status("fetching")
        on_status("analyzing")
        return {"parcel": {"frost_score": 81.2, "confidence": "high"}}, {"years": [2024]}
    return runner


@pytest.fixture
def queue_dir(tmp_path):
    return str(tmp_path / "output")


# -------------------------------------------------------------------- auth
def test_parcel_jobs_need_the_api_token(monkeypatch):
    monkeypatch.delenv("VTA_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as e:
        api._require_token(f"Bearer {TOKEN}")
    assert e.value.status_code == 503            # disabled, not open, when unset
    monkeypatch.setenv("VTA_API_TOKEN", TOKEN)
    for header, code in ((None, 401), ("Basic abc", 401), ("Bearer ", 401),
                         ("Bearer wrong", 403)):
        with pytest.raises(HTTPException) as e:
            api._require_token(header)
        assert e.value.status_code == code, header
    assert api._require_token(f"Bearer {TOKEN}") is None
    assert api._require_token(f"bearer {TOKEN} ") is None


def test_health_reports_the_model_version(monkeypatch):
    monkeypatch.setenv("VTA_API_TOKEN", TOKEN)
    h = api.health()
    assert h["ok"] and h["model_version"] == "v2+z1" and h["frost_model"] == "v2"
    assert h["parcel_jobs"] is True


# --------------------------------------------------------------- the request
def test_requests_carry_structured_fields_only():
    ok = api.ParcelJobRequest(address="1056 Naramata Rd", town="Penticton",
                              listing_acres=9.0, ref="mls-10398703")
    assert ok.model_dump(exclude_none=True)["confirmed"] is False
    for bad in ({"address": "x", "url": "https://evil.example/x"},     # unknown field
                {"address": "x", "title": "free text from an email"},
                {"town": "Oliver"},                                    # nothing to locate
                {"lat": 49.1},                                         # lat without lon
                {"lat": 45.0, "lon": -119.5},                          # far outside BC
                {"address": "x", "buffer_m": 50},
                {"address": "x", "pid": [f"{i:09d}" for i in range(11)]}):
        with pytest.raises(ValidationError):
            api.ParcelJobRequest(**bad)


def test_post_queues_a_job_with_an_unguessable_id(monkeypatch, queue_dir):
    monkeypatch.setenv("VTA_API_TOKEN", TOKEN)
    monkeypatch.setenv("VTA_MAP_BASE_URL", "http://100.71.243.94:8000")
    monkeypatch.setattr(api, "PARCEL_JOBS", jobs.ParcelJobs(queue_dir))
    body = api.ParcelJobRequest(address="1056 Naramata Rd", town="Penticton",
                                ref="mls-1")
    out = api.create_parcel_job(body, _request(), authorization=f"Bearer {TOKEN}")
    assert jobs.JOB_ID_RE.fullmatch(out["job_id"]) and out["status"] == "queued"
    assert out["map_url"] == f"http://100.71.243.94:8000/?job={out['job_id']}"
    st = api.PARCEL_JOBS.read_status(out["job_id"])
    assert st["status"] == "queued" and st["ref"] == "mls-1"
    assert st["request"] == {"address": "1056 Naramata Rd", "town": "Penticton",
                             "ref": "mls-1", "confirmed": False}
    with pytest.raises(HTTPException) as e:
        api.create_parcel_job(body, _request(), authorization=None)
    assert e.value.status_code == 401


def test_get_needs_no_token_but_a_well_formed_existing_id(monkeypatch, queue_dir):
    q = jobs.ParcelJobs(queue_dir)
    monkeypatch.setattr(api, "PARCEL_JOBS", q)
    monkeypatch.delenv("VTA_MAP_BASE_URL", raising=False)
    job_id = q.submit({"address": "1056 Naramata Rd"})
    view = api.get_parcel_job(job_id, _request())
    assert view["status"] == "queued" and view["map_url"].endswith(f"/?job={job_id}")
    assert view["map_url"].startswith("http://frost.test:8000")
    for bad in ("0" * 32, "../../etc", "abc"):
        with pytest.raises(HTTPException) as e:
            api.get_parcel_job(bad, _request())
        assert e.value.status_code == 404


def test_a_full_queue_answers_429(monkeypatch, queue_dir):
    monkeypatch.setenv("VTA_API_TOKEN", TOKEN)
    monkeypatch.setattr(jobs, "QUEUE_MAX", 2)
    monkeypatch.setattr(api, "PARCEL_JOBS", jobs.ParcelJobs(queue_dir))
    body = api.ParcelJobRequest(address="x")
    for _ in range(2):
        api.create_parcel_job(body, _request(), authorization=f"Bearer {TOKEN}")
    with pytest.raises(HTTPException) as e:
        api.create_parcel_job(body, _request(), authorization=f"Bearer {TOKEN}")
    assert e.value.status_code == 429


# --------------------------------------------------------------------- jobs
def test_a_job_walks_resolving_fetching_analyzing_done(queue_dir):
    calls, seen = [], []
    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: _resolved(),
                        runner=_fake_runner(calls))
    real_write = q.write_status

    def spy(job_id, **changes):
        if "status" in changes:
            seen.append(changes["status"])
        return real_write(job_id, **changes)

    q.write_status = spy
    job_id = q.submit({"address": "1056 Naramata Rd", "buffer_m": 1000})
    q.run_job(job_id)
    assert seen == ["queued", "resolving", "fetching", "fetching", "analyzing", "done"]
    st = q.read_status(job_id)
    assert st["metrics"]["frost_score"] == 81.2 and st["provenance"]["years"] == [2024]
    assert st["parcel"]["pid"] == "012-525-499" and "geometry_3005" not in st["parcel"]
    assert calls == [{"pid": "012-525-499", "buffer_m": 1000, "label": "1056 Naramata Rd"}]
    assert st["attempts"] == 1 and st["error"] is None


def test_low_confidence_stops_at_needs_confirmation_with_candidates_on_the_map(queue_dir):
    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: _needs_confirmation(),
                        runner=lambda *a, **k: pytest.fail("scored an unconfirmed parcel"))
    job_id = q.submit({"lat": 49.134041, "lon": -119.582336})
    q.run_job(job_id)
    st = q.read_status(job_id)
    assert st["status"] == "needs_confirmation"
    assert st["parcel"]["candidates"][0]["pid"] == "011-333-596"
    with open(os.path.join(queue_dir, job_id, "parcel.geojson")) as f:
        roles = [ft["properties"]["role"] for ft in json.load(f)["features"]]
    assert roles == ["candidate"]


@pytest.mark.parametrize("exc, message", [
    (parcel.BoundaryError("ring self-intersects"), "bad boundary: ring self-intersects"),
    (ValueError("several PIDs score as one unit only with confirmed: true"), "confirmed"),
    (parcel.ResolveError("ParcelMap BC matched 900 parcels"), "parcel lookup failed"),
])
def test_resolution_errors_fail_the_job_with_the_reason(queue_dir, exc, message):
    def resolver(req):
        raise exc
    q = jobs.ParcelJobs(queue_dir, resolver=resolver, runner=lambda *a, **k: None)
    job_id = q.submit({"geometry": {"type": "Polygon", "coordinates": []}})
    q.run_job(job_id)
    st = q.read_status(job_id)
    assert st["status"] == "failed" and message in st["error"]


def test_an_aoi_outside_the_service_area_fails_before_any_download(queue_dir, monkeypatch):
    from backend.pipeline import tiles
    monkeypatch.setattr(tiles, "load_index", lambda *a, **k: pytest.fail("index loaded"))
    vancouver = _resolved(geometry_3005=box(1210000, 470000, 1210200, 470200))
    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: vancouver, runner=run.run_parcel)
    job_id = q.submit({"pid": "000-000-001"})
    q.run_job(job_id)
    st = q.read_status(job_id)
    assert st["status"] == "failed" and "outside the southern interior" in st["error"]


def test_analysis_errors_fail_the_job(queue_dir):
    def runner(*a, **k):
        raise run.ParcelAnalysisError("cold-air routing (WhiteboxTools) failed")
    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: _resolved(), runner=runner)
    job_id = q.submit({"address": "x"})
    q.run_job(job_id)
    assert "WhiteboxTools" in q.read_status(job_id)["error"]


def test_the_worker_runs_one_job_at_a_time(queue_dir):
    running, peak, done = [0], [0], threading.Event()
    lock = threading.Lock()

    def runner(res, out_dir, **kw):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return {"parcel": {}}, {}

    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: _resolved(), runner=runner)
    ids = [q.submit({"address": f"{i} Road"}) for i in range(4)]
    q.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(q.read_status(i)["status"] == "done" for i in ids):
            done.set()
            break
        time.sleep(0.02)
    assert done.is_set() and peak[0] == 1


def test_restart_requeues_interrupted_jobs_and_fails_twice_interrupted_ones(queue_dir):
    q = jobs.ParcelJobs(queue_dir, resolver=lambda req: _resolved(),
                        runner=_fake_runner([]))
    once = q.submit({"address": "a"})
    twice = q.submit({"address": "b"})
    q.write_status(once, status="analyzing", attempts=1)
    q.write_status(twice, status="fetching", attempts=2)
    restarted = jobs.ParcelJobs(queue_dir, resolver=lambda req: _resolved(),
                                runner=_fake_runner([]))
    restarted.start()
    restarted.queue.join()
    assert restarted.read_status(once)["status"] == "done"
    st = restarted.read_status(twice)
    assert st["status"] == "failed" and "interrupted twice" in st["error"]


def test_old_layers_are_pruned_and_the_json_kept(queue_dir):
    q = jobs.ParcelJobs(queue_dir)
    old, new = q.submit({"address": "a"}), q.submit({"address": "b"})
    for job_id in (old, new):
        os.makedirs(os.path.join(queue_dir, job_id, "layers"))
        with open(os.path.join(queue_dir, job_id, "layers", "frost.png"), "wb") as f:
            f.write(b"png")
        q.write_status(job_id, status="done")
    st = q.read_status(old)
    st["updated_at"] = (datetime.now(timezone.utc) - timedelta(days=31)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(queue_dir, old, "status.json"), "w") as f:
        json.dump(st, f)
    assert q.prune() == [old]
    assert not os.listdir(os.path.join(queue_dir, old, "layers"))
    assert os.listdir(os.path.join(queue_dir, new, "layers")) == ["frost.png"]
    assert q.read_status(old)["layers_pruned"] is True
    assert q.view(old, "http://x")["layers_pruned"] is True
