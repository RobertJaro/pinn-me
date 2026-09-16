#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

data_directory="/glade/work/rjarolim/data/prom3theus/2011_02"
email="robert.jarolim@uni-graz.at"
start="2011-02-14T00:00:00"
end="2011-02-15T00:00:00"


PYTHONPATH=src python -m prom3theus.cli.main download hmi-stokes \
  --output "${data_directory}/hmi/full_disk" \
  --email "${email}" \
  --start "${start}" \
  --end "${end}"

PYTHONPATH=src python -m prom3theus.cli.main download aia-euv \
  --output "${data_directory}/aia/level1" \
  --email "${email}" \
  --start "${start}" \
  --end "${end}" \
  --cadence-seconds 720 \
  --channels 171 193 211
