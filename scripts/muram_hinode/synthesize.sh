#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
case "$*" in
  ""|--overwrite) ;;
  *) echo "Usage: $0 [--overwrite]" >&2; exit 2 ;;
esac

# The same native snapshot 000000 as the HMI experiment; save native spatial resolution.
# Activate the sunerf environment before running; existing outputs are resumed.
PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path /glade/derecho/scratch/rjarolim/muram_snapshot \
  --out "/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_single_frame" \
  --iterations 0 --output-format store \
  --instrument-config configs/muram/hinode/synthesis.yaml "$@"

# AR 11158.
PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path "/glade/campaign/hao/radmhd/Rempel/ASD_2023/Run_Corona_1728x1024x1024_ASD_open/3D" \
  --out "/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_11158" \
  --iterations 1000000 1500000 2000000 --output-format store \
  --instrument-config configs/muram/hinode/synthesis.yaml --batch-size 32 "$@"
