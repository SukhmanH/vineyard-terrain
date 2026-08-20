# Vineyard Terrain Analyzer API.
#
# This app cannot run on a serverless platform: rasterio's bundled GDAL, scipy,
# matplotlib and the WhiteboxTools binary come to roughly 500 MB unpacked, well
# past a serverless function's 250 MB ceiling, and the pipeline writes job
# folders that later requests read back. It wants a container with a disk.
#
# rasterio and scipy ship manylinux wheels with their native libraries bundled,
# so no system GDAL is needed here. libgomp is required by the WhiteboxTools
# binary, which is OpenMP-linked.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    VTA_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the WhiteboxTools binary at build time. Left to first request it would
# download mid-upload, adding a minute to that request and failing outright on
# a host with no outbound network.
RUN python -c "import whitebox; whitebox.WhiteboxTools()" || \
    echo "WhiteboxTools prefetch failed; flow accumulation will degrade to null"

COPY backend/ backend/
COPY frontend/ frontend/

# Job output and uploads live on a mounted volume so they survive a restart.
RUN mkdir -p /data/uploads /data/output
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
