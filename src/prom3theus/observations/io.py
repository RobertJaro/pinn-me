"""Deterministic, bounded parallel I/O used by observation adapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from astropy.io.fits.verify import VerifyWarning
from tqdm.auto import tqdm


def ordered_parallel_map(
    function,
    items,
    workers: int,
    *,
    progress: bool = False,
    description: str | None = None,
    unit: str = "file",
):
    items = list(items)
    if not items:
        return []
    workers = int(workers)
    if workers < 1:
        raise ValueError("I/O worker count must be positive.")
    if workers == 1 or len(items) == 1:
        iterator = tqdm(
            items,
            total=len(items),
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not progress,
        )
        return [function(item) for item in iterator]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = {
            executor.submit(function, item): index for index, item in enumerate(items)
        }
        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not progress,
        )
        for future in iterator:
            results[futures[future]] = future.result()
    return results


def ordered_parallel_fits_map(function, items, workers: int, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        return ordered_parallel_map(function, items, workers, **kwargs)


__all__ = ["ordered_parallel_fits_map", "ordered_parallel_map"]
