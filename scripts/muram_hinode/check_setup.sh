#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

PYTHONPATH=src python -m prom3theus.muram.cli validate-config configs/muram/hinode/mhs.yaml
PYTHONPATH=src python scripts/muram_hinode/check_setup.py
