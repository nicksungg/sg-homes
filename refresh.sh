#!/usr/bin/env bash
# Rebuild everything: fresh HDB + URA data -> features -> models -> map data.
set -euo pipefail
cd "$(dirname "$0")"
rm -f data/raw/resale_2017_onwards.csv  # the only dataset updated monthly
.venv/bin/python pipeline/download.py
.venv/bin/python pipeline/geocode.py   # OneMap results are cached; reruns are fast
.venv/bin/python pipeline/features.py
.venv/bin/python pipeline/train.py
.venv/bin/python pipeline/forecast.py
(cd pipeline && ../.venv/bin/python score.py)
.venv/bin/python pipeline/ura_download.py       # new dated snapshot, merged into the archive
(cd pipeline && ../.venv/bin/python private.py) # condo + landed: features, train, export
echo "done. (re)start the server: cd pipeline && ../.venv/bin/python server.py 8765"
