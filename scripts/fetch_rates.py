"""Fetch all FRED series and the ECB €STR history into data/raw/."""

from __future__ import annotations

import requests

from fxvrp.config import load_config
from fxvrp.data import ecb, fred
from fxvrp.log import get_logger

logger = get_logger("scripts.fetch_rates")


def main() -> None:
    config = load_config()
    session = requests.Session()

    for series_id in config.fred.series:
        frame = fred.fetch_series(
            session,
            config.fred.base_url,
            series_id,
            timeout=config.cboe.request_timeout_s,
        )
        path = fred.save_series(frame, series_id, config.paths.raw_dir)
        logger.info("saved %s -> %s", series_id, path)

    estr = ecb.fetch_estr(session, config.ecb, timeout=config.cboe.request_timeout_s)
    path = ecb.save_estr(estr, config.paths.raw_dir)
    logger.info("saved ESTR -> %s", path)


if __name__ == "__main__":
    main()
