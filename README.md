# Vineyard Terrain Analyzer

Upload a LidarBC 1 m DEM tile, get vineyard-siting terrain analysis over a
satellite basemap with an opacity slider. Read CLAUDE.md for the full spec,
build phases, and gotchas. Quick start:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn backend.app:app --reload --port 8000
    # open http://localhost:8000

CLI without the server:

    python -m backend.pipeline.run path/to/tile.tif output/test1
