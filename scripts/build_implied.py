"""Compute MFIV indices from every chain snapshot on disk.

For `_SPX` the output is a VIX replication, compared against published VIXCLS
(the ADR 0002 validation); for FXE it *is* the continuation of the
decommissioned EVZ. Writes data/processed/implied_daily.parquet and the
reconciliation report docs/reports/vix_replication.md. Rerunning extends both
as the daily scraper accrues snapshots.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import polars as pl

from fxvrp.config import Config, load_config
from fxvrp.data.rates import cc_rate_from_fed_funds, rate_asof
from fxvrp.implied import (
    interpolate_constant_maturity,
    select_term_expiries,
    single_expiry_variance,
)
from fxvrp.implied.mfiv import expiry_settlement, index_level, minutes_to
from fxvrp.log import get_logger

logger = get_logger("scripts.build_implied")

MFIV_SYMBOLS = ("fxe", "spx")  # snapshot directory names (underscores stripped)


def snapshot_index(config: Config, symbol_dir: str) -> list[Path]:
    base = config.paths.raw_dir / "chains" / symbol_dir
    return sorted(base.glob("*.parquet")) if base.exists() else []


def _dominant_root(expiry_frame: pl.DataFrame) -> str:
    counts = (
        expiry_frame.filter((pl.col("bid") > 0) & (pl.col("ask") > pl.col("bid")))
        .group_by("root")
        .len()
        .sort("len", descending=True)
    )
    root = counts["root"][0] if counts.height else expiry_frame["root"][0]
    return str(root)


def compute_snapshot_index(
    chain: pl.DataFrame,
    config: Config,
    dff: pl.DataFrame,
    underlying_yield: float | None,
) -> dict[str, object] | None:
    """MFIV index for one snapshot.

    ``underlying_yield`` (cc): when known (FXE: the EUR overnight rate the
    trust distributes), the carry forward S e^{(r-q)T} backstops expiries where
    no strike is two-sided on both sides — a real feature of thin ETF chains.
    """
    quote_time = chain["quote_time"][0]
    if not isinstance(quote_time, datetime):
        logger.warning("snapshot missing quote_time; skipped")
        return None
    asof_date = chain["quote_time"].dt.convert_time_zone(config.implied.settlement_tz)[0]
    r = cc_rate_from_fed_funds(rate_asof(dff, quote_time.date()))
    spot = chain["spot"][0]

    minutes_by_expiry: dict[date, float] = {}
    root_by_expiry: dict[date, str] = {}
    for (expiry,), expiry_frame in chain.group_by("expiry"):
        assert isinstance(expiry, date)
        root = _dominant_root(expiry_frame)
        settlement = expiry_settlement(root, expiry, config.implied)
        minutes = minutes_to(quote_time, settlement)
        if minutes > 0:
            minutes_by_expiry[expiry] = minutes
            root_by_expiry[expiry] = root

    # compute a strip for every eligible expiry; term selection then runs over
    # *usable* expiries only (near-term FXE routinely has no OTM bids at all)
    slices = {}
    for expiry in sorted(minutes_by_expiry):
        frame = chain.filter(
            (pl.col("expiry") == expiry) & (pl.col("root") == root_by_expiry[expiry])
        ).select("strike", "call_put", "bid", "ask")
        t_years = minutes_by_expiry[expiry] / (365.0 * 24.0 * 60.0)
        fallback = (
            float(spot) * math.exp((r - underlying_yield) * t_years)
            if underlying_yield is not None and spot is not None
            else None
        )
        try:
            candidate = single_expiry_variance(
                frame, t_years=t_years, r=r, cfg=config.implied, fallback_forward=fallback
            )
        except ValueError as error:
            logger.info("expiry %s unusable: %s", expiry, error)
            continue
        if candidate.n_options < config.implied.min_strip_strikes:
            logger.info("expiry %s unusable: only %d strip strikes", expiry, candidate.n_options)
            continue
        slices[expiry] = candidate

    near, nxt = select_term_expiries(
        {expiry: minutes_by_expiry[expiry] for expiry in slices}, config.implied
    )

    var30 = interpolate_constant_maturity(
        slices[near].sigma_sq,
        minutes_by_expiry[near],
        slices[nxt].sigma_sq,
        minutes_by_expiry[nxt],
        config.implied.target_days,
    )
    return {
        "symbol": str(chain["underlying"][0]),
        "date": asof_date.date() if isinstance(asof_date, datetime) else quote_time.date(),
        "index": index_level(var30),
        "variance_30d": var30,
        "near_expiry": near,
        "next_expiry": nxt,
        "near_days": minutes_by_expiry[near] / (24.0 * 60.0),
        "next_days": minutes_by_expiry[nxt] / (24.0 * 60.0),
        "n_options_near": slices[near].n_options,
        "n_options_next": slices[nxt].n_options,
        "rate_cc": r,
    }


def main() -> None:
    config = load_config()
    dff = pl.read_parquet(config.paths.raw_dir / "fred" / "DFF.parquet")
    estr = pl.read_parquet(config.paths.raw_dir / "ecb" / "estr.parquet").rename({"rate": "value"})

    rows: list[dict[str, object]] = []
    unusable: dict[str, int] = {}
    for symbol_dir in MFIV_SYMBOLS:
        for path in snapshot_index(config, symbol_dir):
            chain = pl.read_parquet(path)
            if chain.height == 0:
                continue
            # FXE's distribution yield is the EUR overnight rate (ACT/360, like DFF)
            underlying_yield = (
                cc_rate_from_fed_funds(rate_asof(estr, date.fromisoformat(path.stem)))
                if symbol_dir == "fxe"
                else None
            )
            try:
                row = compute_snapshot_index(chain, config, dff, underlying_yield)
            except ValueError:
                logger.exception("MFIV failed for %s", path)
                unusable[symbol_dir] = unusable.get(symbol_dir, 0) + 1
                continue
            if row is not None:
                rows.append(row)

    if not rows:
        logger.warning("no snapshots on disk; nothing to do")
        return
    result = pl.DataFrame(rows).sort("symbol", "date")

    out = config.paths.processed_dir / "implied_daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out)

    # reconciliation against published VIX
    vix = (
        pl.read_parquet(config.paths.raw_dir / "fred" / "VIXCLS.parquet")
        .rename({"value": "vix_published"})
        .drop_nulls("vix_published")
    )
    spx = (
        result.filter(pl.col("symbol") == "_SPX")
        .join(vix, on="date", how="left")
        .with_columns((pl.col("index") - pl.col("vix_published")).alias("diff"))
    )
    matched = spx.drop_nulls("vix_published")

    report_dir = config.paths.reports_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VIX replication report\n",
        "Generated by `scripts/build_implied.py`; extends daily with the scraper.\n",
        f"\nSnapshots processed: {result.height} "
        f"(SPX: {spx.height}, FXE: {result.filter(pl.col('symbol') == 'FXE').height})\n",
        "\n## Replicated vs published VIX\n\n",
        "| date | replicated | published | diff | near/next days | strikes used |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for row in spx.iter_rows(named=True):
        pub = f"{row['vix_published']:.2f}" if row["vix_published"] is not None else "n/a"
        diff = f"{row['diff']:+.2f}" if row["diff"] is not None else "n/a"
        lines.append(
            f"| {row['date']} | {row['index']:.2f} | {pub} | {diff} "
            f"| {row['near_days']:.1f}/{row['next_days']:.1f} "
            f"| {row['n_options_near']}+{row['n_options_next']} |\n"
        )
    if matched.height:
        mad = float(matched["diff"].abs().mean())  # type: ignore[arg-type]
        lines.append(
            f"\nMean absolute deviation over {matched.height} matched days: "
            f"**{mad:.2f} vol points**\n"
        )
    lines.append("\n## FXE (EVZ continuation)\n\n| date | index | strikes used |\n|---|---|---|\n")
    for row in result.filter(pl.col("symbol") == "FXE").iter_rows(named=True):
        lines.append(
            f"| {row['date']} | {row['index']:.2f} "
            f"| {row['n_options_near']}+{row['n_options_next']} |\n"
        )
    if unusable:
        lines.append(
            "\nSnapshots with no usable strip (zero-bid OTM wings; see ADR 0002 "
            f"addendum): {unusable}\n"
        )
    (report_dir / "vix_replication.md").write_text("".join(lines), encoding="utf-8")
    logger.info("wrote %s and %s", out, report_dir / "vix_replication.md")


if __name__ == "__main__":
    main()
