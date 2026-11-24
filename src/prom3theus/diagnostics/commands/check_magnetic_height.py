"""Check radial magnetic constancy and retained temperature stratification in a P3S."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--heights-megameter", type=float, nargs="+", default=(0, .1, .15, .3, 1))
    args = parser.parse_args()

    import torch
    from prom3theus.diagnostics.magnetic_height import check_magnetic_height

    torch.set_num_threads(4)
    result = check_magnetic_height(args.source, args.output_directory,
                                   sample_count=args.sample_count,
                                   heights_megameter=args.heights_megameter)
    print(result["paths"]["report"])
    print(f"Height-independent parameterization checks passed: {result['report']['passed']}")
    return 0 if result["report"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
