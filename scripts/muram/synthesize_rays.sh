#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

# Independent output from column synthesis. Edit placement for the actual box.
# Here the bottom cell center is explicitly placed at the nominal solar radius.
PYTHONPATH=src python -m prom3theus.muram.synthesize_rays \
  --source-path /glade/derecho/scratch/rjarolim/muram_snapshot \
  --out /glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_rays \
  --instrument-config configs/muram/hinode/synthesis.yaml \
  --iterations 0 \
  --bottom-height-megameters 0 \
  --volume-height-megameters 3 \
  --depth-samples 128 \
  --batch-size 32 \
  --device auto
