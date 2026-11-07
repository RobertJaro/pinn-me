#!/usr/bin/env python3
"""Unpack MURaM snapshot archives into a local scratch directory."""

import shutil
from pathlib import Path

SOURCE_DIR = Path("/glade/campaign/hao/radmhd/Rempel/Flare_chianti_phot_v300_va900_hpm_192x64_inv_pol/3D")
TARGET_DIR = Path("/glade/derecho/scratch/rjarolim/muram_snapshot")
ITERATIONS = (0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000)

TARGET_DIR.mkdir(parents=True, exist_ok=True)
for iteration in ITERATIONS:
    archive = SOURCE_DIR / f"{iteration:06d}.tar"
    print(f"Unpacking {archive} -> {TARGET_DIR}")
    shutil.unpack_archive(archive, TARGET_DIR)
