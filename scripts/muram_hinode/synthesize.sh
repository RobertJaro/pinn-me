#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

bash scripts/muram/synthesize_snapshots.sh

# AR 11158. The photosphere lies about 7.5 Mm above the bottom native cell.
PYTHONPATH=src python -m prom3theus.muram.cli synthesize \
  --source-path "/glade/campaign/hao/radmhd/Rempel/ASD_2023/Run_Corona_1728x1024x1024_ASD_open/3D" \
  --out "/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_11158_uniform32" \
  --iterations 1000000 1500000 2000000 --output-format store \
  --instrument-config configs/muram/hinode/synthesis.yaml --batch-size 128 \
  --inversion-depth-grid configs/muram/hinode/mhs.yaml --photosphere-height-megameters 7.5

# AR 11158: ray synthesis viewed from above at disk center.
# Register the mean photosphere approximately at the nominal solar radius.
PYTHONPATH=src python -m prom3theus.muram.synthesize_rays \
  --source-path "/glade/campaign/hao/radmhd/Rempel/ASD_2023/Run_Corona_1728x1024x1024_ASD_open/3D" \
  --out "/glade/derecho/scratch/rjarolim/muram_synthetic_observations/hinode_11158_rays_12Mm" \
  --iterations 1000000 1500000 2000000 \
  --instrument-config configs/muram/hinode/synthesis.yaml \
  --longitude-deg 0 --latitude-deg 0 \
  --observer-direction 1 0 0 \
  --bottom-height-megameters -7.5 \
  --volume-height-megameters 12 \
  --depth-samples 256 \
  --batch-size 128 \
  --device cuda
