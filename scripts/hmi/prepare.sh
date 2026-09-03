#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-subframes data/hmi/2024-03-23/full_disk \
  --output data/hmi/2024-03-23/subframe \
  --longitude-deg 215 \
  --latitude-deg -12 \
  --width-pixels 1024 \
  --height-pixels 512

PYTHONPATH=src python -m prom3theus.cli.main prepare hmi-responses data/hmi/2024-03-23/subframe \
  --output data/calibration/hmi/2024-03-23 \
  --email "${JSOC_EMAIL:?Set JSOC_EMAIL to your registered JSOC email address}" \
  --phase-map-fsn 230562565
