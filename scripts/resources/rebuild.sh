#!/bin/bash -l

set -euo pipefail
cd "$(dirname "$0")/../.."
test "$#" -eq 0

python -m resource_builder.build \
  --output-directory build/reproduced-lte-resources
