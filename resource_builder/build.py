"""Reproduce the complete packaged LTE resource bundle from pinned inputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path

from . import common_atomic, hinode_sp, hmi_stokes
from ._shared import (
    LOGGER,
    PACKAGED_ROOT,
    _default_cache_directory,
    _fetch_inputs,
    _read_json,
    _seal_bundle,
    _validate_reproduction,
)


INSTRUMENT_BUILDERS = (hinode_sp, hmi_stokes)


def reproduce(output: Path, cache_directory: Path, workers: int) -> None:
    """Build common and registered instrument resources, then verify exactly."""

    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = deepcopy(_read_json(PACKAGED_ROOT / "sources.json"))
    payloads = _fetch_inputs(source_manifest, cache_directory, workers)
    stic_table = common_atomic.build(output, payloads, source_manifest)
    for instrument_builder in INSTRUMENT_BUILDERS:
        instrument_builder.build(output, payloads, source_manifest)
    _seal_bundle(output, source_manifest, stic_table)
    _validate_reproduction(output, source_manifest)
    LOGGER.info("Exact LTE resource reproduction verified: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--cache-directory", type=Path, default=_default_cache_directory()
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reproduce(
        args.output_directory.expanduser().resolve(),
        args.cache_directory.expanduser().resolve(),
        args.workers,
    )


if __name__ == "__main__":
    main()
