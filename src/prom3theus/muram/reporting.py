"""CLI logging and terminal-aware progress without changing library log handlers."""

from contextlib import contextmanager
import logging
import sys
from time import gmtime, perf_counter

from tqdm.auto import tqdm

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ProgressLogHandler(logging.StreamHandler):
    """Write above active tqdm bars instead of corrupting their display."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


@contextmanager
def cli_logging(level="INFO"):
    """Configure only this command's package logger and restore it afterwards."""
    logger = logging.getLogger("prom3theus.muram")
    previous_level, previous_propagate = logger.level, logger.propagate
    handler = ProgressLogHandler(sys.stderr)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise
    except Exception as error:
        logger.error("%s: %s", type(error).__name__, error)
        logger.debug("Command traceback", exc_info=True)
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def progress_enabled(enabled):
    return bool(enabled and sys.stderr.isatty())


def progress_bar(iterable=None, *, total=None, description, unit, enabled=True):
    """Use counts, elapsed time, throughput and ETA; keep redirected logs plain."""
    return tqdm(iterable, total=total, desc=description, unit=unit, file=sys.stderr,
                disable=not progress_enabled(enabled), dynamic_ncols=True,
                mininterval=1.0, leave=False)


@contextmanager
def timed_stage(logger, message, *args):
    """Log stage boundaries, including elapsed time and failure context."""
    label = message % args if args else message
    started = perf_counter()
    logger.info("%s", label)
    try:
        yield
    except BaseException as error:
        logger.error("%s failed after %.1f s (%s: %s)", label,
                     perf_counter() - started, type(error).__name__, error)
        raise
    else:
        logger.info("%s finished in %.1f s", label, perf_counter() - started)
