"""Ordered preparation work with shared-memory workers and completion progress."""

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os


def parallel_map(function, items, *, description):
    """Share inputs between threads; return results in deterministic input order.

    Limit full-disk working memory with PROM3THEUS_PREP_WORKERS (default 16).
    Exceptions propagate only after running workers finish, allowing safe cleanup.
    """
    from tqdm.auto import tqdm

    items = list(items)
    workers = int(os.environ.get("PROM3THEUS_PREP_WORKERS", "16"))
    if workers < 1:
        raise ValueError("PROM3THEUS_PREP_WORKERS must be positive.")
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        remaining = iter(enumerate(items))
        futures = {}

        def submit_next():
            entry = next(remaining, None)
            if entry is not None:
                index, item = entry
                futures[pool.submit(function, item)] = index

        for _ in range(min(workers, len(items))):
            submit_next()
        try:
            with tqdm(total=len(items), desc=description, unit="item") as progress:
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        results[futures.pop(future)] = future.result()
                        progress.update()
                        submit_next()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results
