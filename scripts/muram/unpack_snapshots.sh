#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 0

source_dir=/glade/campaign/hao/radmhd/Rempel/Flare_chianti_phot_v300_va900_hpm_192x64_inv_pol/3D
target_dir=/glade/derecho/scratch/rjarolim/muram_snapshot
mkdir -p "$target_dir"
echo "Unpacking $source_dir/000000.tar into $target_dir"
tar -xf "$source_dir/000000.tar" -C "$target_dir"
