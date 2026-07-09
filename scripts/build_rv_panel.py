"""Build the daily realised-variance panel from all ingested ticks.

Writes data/processed/rv_daily.parquet. Rerunning extends the panel as the tick
backfill progresses; the panel range defaults to the config ingestion window.
"""

from __future__ import annotations

import argparse
from datetime import date

from fxvrp.config import load_config
from fxvrp.log import get_logger
from fxvrp.realized.panel import build_panel

logger = get_logger("scripts.build_rv_panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()

    config = load_config()
    start = args.start or config.dukascopy.start_date
    end = args.end or config.dukascopy.end_date

    result = build_panel(config.paths.raw_dir, config.dukascopy, config.realized, start, end)
    out_path = config.paths.processed_dir / "rv_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.frame.write_parquet(out_path)
    logger.info("wrote %s: %d days (%d thin)", out_path, result.n_days, result.n_thin)


if __name__ == "__main__":
    main()
