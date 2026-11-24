"""Exclusive run directory, progress logs and atomic artifact writes."""

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger("prom3theus.surrogate")


@contextmanager
def open_run(directory: Path):
    """Hold the training lock and close both log handlers on every exit."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".trainer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handlers = [
            logging.FileHandler(directory / "train.log"),
            logging.StreamHandler(),
        ]
        level = logger.level
        propagate = logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in handlers:
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(handler)
        try:
            yield
        except Exception:
            logger.exception("Mapper training failed")
            raise
        finally:
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
            logger.setLevel(level)
            logger.propagate = propagate


def atomic_save(path: Path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def write_metrics(directory: Path, stage: str, **values):
    record = json.dumps({"stage": stage, **values}, allow_nan=False)
    with (directory / "metrics.jsonl").open("a") as file:
        file.write(record + "\n")
