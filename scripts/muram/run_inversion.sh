#!/bin/bash -l

#PBS -N prom3theus-muram
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

set -euo pipefail
test "$#" -eq 0
cd "${PBS_O_WORKDIR:-$(dirname "$0")/../..}"

# Prepare the Hinode store with scripts/muram/synthesize_ar11158.pbs.
# Submit scripts/muram_hinode/run.sh separately for the MHS inversion.
PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/stokes_only.yaml

PYTHONPATH=src python -m prom3theus.muram.cli invert configs/muram/hinode/mhs.yaml