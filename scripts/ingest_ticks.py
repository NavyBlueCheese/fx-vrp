"""Dukascopy tick ingestion driver.

Resumable by construction: days already on disk are skipped, so interrupting and
rerunning is always safe. `--sample` ingests the bounded config window used for
smoke checks; the default is the full 2007→2025 history (long — run unattended).
"""

from __future__ import annotations

import argparse
from datetime import date

import requests

from fxvrp.config import load_config
from fxvrp.data.dukascopy import fx_days, ingest_day
from fxvrp.log import get_logger

logger = get_logger("scripts.ingest_ticks")

_PROGRESS_EVERY = 20  # days between progress lines; cosmetic only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="config sample window only")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    config = load_config()
    cfg = config.dukascopy
    start = args.start or (cfg.sample_start_date if args.sample else cfg.start_date)
    end = args.end or (cfg.sample_end_date if args.sample else cfg.end_date)

    session = requests.Session()
    n_days = n_skipped = n_ticks = 0
    for day in fx_days(start, end):
        result = ingest_day(session, cfg, day, config.paths.raw_dir)
        n_days += 1
        if result.skipped:
            n_skipped += 1
        else:
            n_ticks += result.n_ticks
        if n_days % _PROGRESS_EVERY == 0:
            logger.info(
                "progress: %d days processed (%d skipped), %d new ticks",
                n_days,
                n_skipped,
                n_ticks,
            )

    logger.info(
        "ingestion complete %s..%s: %d days, %d already on disk, %d new ticks",
        start,
        end,
        n_days,
        n_skipped,
        n_ticks,
    )


if __name__ == "__main__":
    main()
