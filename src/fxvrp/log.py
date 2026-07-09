"""Logging helpers.

Every data filter in the project reports rows in, rows out, and the reason through
``log_filter`` so that any processed dataset can be reconciled with its raw source.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the project namespace, configuring output once."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT)
    return logging.getLogger(f"fxvrp.{name}")


def log_filter(logger: logging.Logger, step: str, n_in: int, n_out: int, reason: str) -> None:
    """Record a row-dropping step; nothing in the pipeline drops rows silently."""
    logger.info(
        "filter %-28s rows_in=%-9d rows_out=%-9d dropped=%-8d reason=%s",
        step,
        n_in,
        n_out,
        n_in - n_out,
        reason,
    )
