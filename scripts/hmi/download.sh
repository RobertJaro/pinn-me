#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.cli.main download hmi-stokes \
  --output data/hmi/2024-03-23/full_disk \
  --email "${JSOC_EMAIL:?Set JSOC_EMAIL to your registered JSOC email address}" \
  --start 2024-03-23T22:12:00 \
  --end 2024-03-24T02:12:00
