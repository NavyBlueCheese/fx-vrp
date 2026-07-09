"""The volatility signature plot from real EURUSD ticks.

Averages RV(Δ) across all FX days currently on disk and writes
paper/figures/signature_plot.pdf (+ the underlying CSV next to it, so the
figure is reproducible and auditable).
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from fxvrp.config import load_config
from fxvrp.log import get_logger
from fxvrp.realized.panel import _load_window_ticks
from fxvrp.realized.signature import average_signature, signature_curve

logger = get_logger("scripts.fig_signature")

_ANNUALIZE_DAYS = 252  # display-only annualisation of daily RV for the y-axis
_PCT = 100.0
_SATURDAY = 5  # date.weekday(): Monday=0 .. Saturday=5


def main() -> None:
    config = load_config()
    tick_dir = config.paths.raw_dir / "ticks" / config.dukascopy.instrument.lower()
    curves: list[pl.DataFrame] = []
    for path in sorted(tick_dir.glob("*.parquet")):
        calendar_day = date.fromisoformat(path.stem)
        label = calendar_day + timedelta(days=1)  # close label following this file
        if label.weekday() >= _SATURDAY:
            continue
        ticks = _load_window_ticks(config.paths.raw_dir, config.dukascopy.instrument, label)
        if ticks is None or ticks.height == 0:
            continue
        curves.append(signature_curve(ticks, label, config.realized))

    if not curves:
        logger.warning("no tick data on disk; nothing to plot")
        return

    table = average_signature(curves)
    out_dir = config.root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_dir / "signature_plot.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    vol = (table["mean_rv"] * _ANNUALIZE_DAYS).sqrt() * _PCT
    ax.plot(table["interval_s"], vol, marker="o", color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("sampling interval (seconds, log scale)")
    ax.set_ylabel("annualised volatility from mean RV (%)")
    ax.set_title(f"EURUSD volatility signature ({len(curves)} days, mid-quotes, previous-tick)")
    ax.axvline(config.realized.grid_interval_s, color="grey", linestyle="--", linewidth=0.9)
    ax.annotate(
        "baseline 5-min",
        xy=(config.realized.grid_interval_s, float(vol.min())),
        fontsize=9,
        color="grey",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "signature_plot.pdf")
    logger.info("wrote %s (%d days averaged)", out_dir / "signature_plot.pdf", len(curves))


if __name__ == "__main__":
    main()
