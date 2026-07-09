"""Daily CBOE chain snapshot: FXE (the object of study), _SPX (VIX validation),
and the minor currency ETFs.

Idempotent per (symbol, quote date): a snapshot already on disk is not re-fetched,
so the scheduled task can fire repeatedly without duplicating data. The quote date
is the CBOE quote timestamp's US/Eastern date — not the local machine date — so a
post-close scrape run early the next morning in another timezone files under the
correct trading day.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import requests

from fxvrp.config import load_config
from fxvrp.data import cboe
from fxvrp.log import get_logger

logger = get_logger("scripts.scrape_chains")

_EASTERN = ZoneInfo("America/New_York")


def main() -> None:
    config = load_config()
    session = requests.Session()
    n_saved = 0

    for symbol in config.cboe.symbols:
        try:
            payload = cboe.fetch_chain(
                session,
                config.cboe.base_url,
                symbol,
                timeout=config.cboe.request_timeout_s,
            )
            fetched_at = datetime.now(tz=UTC)
            frame = cboe.parse_chain(payload, symbol, fetched_at)

            quote_time = frame["quote_time"][0] if frame.height else None
            snapshot_date = (
                quote_time.astimezone(_EASTERN).date()
                if quote_time is not None
                else fetched_at.astimezone(_EASTERN).date()
            )

            json_path, _ = cboe.snapshot_paths(config.paths.raw_dir, symbol, snapshot_date)
            if json_path.exists():
                logger.info("%s %s already on disk; skipping", symbol, snapshot_date)
                continue

            cboe.save_snapshot(payload, frame, config.paths.raw_dir, symbol, snapshot_date)
            two_sided = frame.filter((frame["bid"] > 0) & (frame["ask"] > frame["bid"])).height
            logger.info(
                "%s %s: %d contracts, %d two-sided, spot=%s",
                symbol,
                snapshot_date,
                frame.height,
                two_sided,
                frame["spot"][0] if frame.height else "n/a",
            )
            n_saved += 1
        except Exception:
            # one symbol failing must not lose the day's other snapshots
            logger.exception("scrape failed for %s", symbol)
        time.sleep(config.cboe.throttle_s)

    logger.info("scrape complete: %d snapshots saved", n_saved)


if __name__ == "__main__":
    main()
