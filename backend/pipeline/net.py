"""Outbound HTTP for the parcel path. Every request the server makes goes through here.

The parcel API makes the server download data on someone else's behalf, so where it may go
is pinned: the LidarBC object store, the DataBC WFS and the BC Address Geocoder, and never a
URL taken from a request. Both rules live here rather than at each call site, and redirects
are refused so an allowed host cannot bounce a request somewhere else. The one other way out
is GDAL /vsicurl/ on bucket keys from our own listing (tiles.remote_path).
"""
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request

BUCKET = "https://nrs.objectstore.gov.bc.ca/gdwuts"
WFS = "https://openmaps.gov.bc.ca/geo/pub/wfs"
GEOCODER = "https://geocoder.api.gov.bc.ca/addresses.json"
ALLOWED_HOSTS = frozenset(urllib.parse.urlsplit(u).hostname
                          for u in (BUCKET, WFS, GEOCODER))

TIMEOUT_S = 60
MAX_RESPONSE_BYTES = 32 * 1024 * 1024   # a page of parcel polygons is far below this
CHUNK = 1 << 20
USER_AGENT = "vineyard-terrain/parcel-frost"


class NetError(RuntimeError):
    """An outbound request failed or was refused."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise NetError(f"refused a {code} redirect to {newurl}")


_opener = urllib.request.build_opener(_NoRedirect)


def check_url(url):
    """Return url if it is https to an allowed host, else raise NetError."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise NetError(f"outbound host not allowed: {parts.scheme}://{parts.hostname}")
    return url


def _host(url):
    return urllib.parse.urlsplit(url).hostname


def _request(url):
    return urllib.request.Request(check_url(url), headers={"User-Agent": USER_AGENT})


def get(url, params=None, timeout=TIMEOUT_S, max_bytes=MAX_RESPONSE_BYTES):
    """GET url (plus query params) and return the body, at most max_bytes."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = _request(url)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
    except NetError:
        raise
    except urllib.error.HTTPError as exc:
        raise NetError(f"HTTP {exc.code} from {_host(url)}") from exc
    except (urllib.error.URLError, OSError) as exc:   # DNS, resets, timeouts
        raise NetError(f"{_host(url)} unreachable: {exc}") from exc
    if len(data) > max_bytes:
        raise NetError(f"response from {_host(url)} is over {max_bytes} bytes")
    return data


def get_json(url, params=None, **kw):
    return json.loads(get(url, params, **kw))


def download(url, dest, max_bytes, expect_size=None, expect_md5=None,
             timeout=TIMEOUT_S):
    """Stream url to dest. The file only appears at dest once it is complete and matches
    the expected size and MD5, so a dropped connection never leaves a truncated tile in
    the cache. Returns the number of bytes written."""
    req = _request(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    md5, n = hashlib.md5(), 0
    try:
        with _opener.open(req, timeout=timeout) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                n += len(chunk)
                if n > max_bytes:
                    raise NetError(f"{url} is over {max_bytes} bytes")
                md5.update(chunk)
                out.write(chunk)
        if expect_size is not None and n != expect_size:
            raise NetError(f"short download of {url}: {n} of {expect_size} bytes")
        if expect_md5 is not None and md5.hexdigest() != expect_md5:
            raise NetError(f"checksum mismatch downloading {url}")
        os.replace(part, dest)
    except NetError:
        _remove(part)
        raise
    except urllib.error.HTTPError as exc:
        _remove(part)
        raise NetError(f"HTTP {exc.code} from {_host(url)}") from exc
    except (urllib.error.URLError, OSError) as exc:
        _remove(part)
        raise NetError(f"{_host(url)} unreachable: {exc}") from exc
    return n


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
