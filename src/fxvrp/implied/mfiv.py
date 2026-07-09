"""Model-free implied variance — the CBOE VIX methodology.

Single-expiry variance (CBOE VIX White Paper, 2022 revision; Demeterfi, Derman,
Kamal & Zou 1999 for the log-contract foundation):

    σ² = (2/T) Σᵢ (ΔKᵢ/Kᵢ²) e^{rT} Q(Kᵢ)  −  (1/T) (F/K₀ − 1)²,

    F  = K* + e^{rT} (C(K*) − P(K*)),  K* = argmin |C(K) − P(K)|,
    K₀ = largest strike ≤ F,
    Q  = OTM mid (puts below K₀, calls above, the average of both at K₀),
    ΔKᵢ = (Kᵢ₊₁ − Kᵢ₋₁)/2, one-sided at the ends.

Strike selection follows the white paper exactly: only two-sided quotes enter;
a single zero-bid strike is skipped; a wing is truncated after
`zero_bid_consecutive_stop` (=2) consecutive zero bids.

Constant-maturity interpolation in *minutes*:

    σ²₃₀ = [T₁σ₁² (N₂−N₃₀)/(N₂−N₁) + T₂σ₂² (N₃₀−N₁)/(N₂−N₁)] · N₃₆₅/N₃₀,
    index = 100 √σ²₃₀.

When both available expiries sit on one side of 30 days the same weights
extrapolate linearly (CBOE's own fallback). Settlement conventions: PM-settled
roots (SPXW, equity/ETF options) expire 16:00 ET; standard third-Friday index
options (root SPX) are AM-settled at 09:30 ET.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from fxvrp.config import ImpliedConfig
from fxvrp.log import get_logger

logger = get_logger("implied.mfiv")

MINUTES_PER_YEAR = 525_600  # 365-day basis, per the white paper
_THIRD_FRIDAY_RANGE = range(15, 22)
_FRIDAY = 4
_MIN_PARITY_PAIRS = 1
_MIN_TERM_EXPIRIES = 2


@dataclass(frozen=True)
class ExpiryVariance:
    sigma_sq: float  # annualised variance attributed to this expiry
    forward: float
    k0: float
    n_options: int  # strikes entering the sum
    t_years: float


def expiry_settlement(root: str, expiry: date, cfg: ImpliedConfig) -> datetime:
    """Settlement instant of a contract, in UTC."""
    tz = ZoneInfo(cfg.settlement_tz)
    is_standard_index = (
        root == "SPX" and expiry.day in _THIRD_FRIDAY_RANGE and expiry.weekday() == _FRIDAY
    )
    local = (
        time.fromisoformat(cfg.settlement_standard_local)
        if is_standard_index
        else time.fromisoformat(cfg.settlement_weekly_local)
    )
    return datetime.combine(expiry, local, tzinfo=tz).astimezone(UTC)


def minutes_to(asof: datetime, settlement: datetime) -> float:
    """Minutes between two aware instants."""
    if asof.tzinfo is None or settlement.tzinfo is None:
        raise ValueError("asof and settlement must be timezone-aware")
    return (settlement - asof).total_seconds() / 60.0


def _two_sided(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter((pl.col("bid") > 0.0) & (pl.col("ask") > pl.col("bid"))).with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid")
    )


def _forward_from_parity(frame: pl.DataFrame, t_years: float, r: float) -> float:
    """F = K* + e^{rT}(C − P) at the strike minimising |C − P| (two-sided pairs)."""
    quotes = _two_sided(frame)
    pairs = (
        quotes.filter(pl.col("call_put") == "C")
        .select("strike", pl.col("mid").alias("call_mid"))
        .join(
            quotes.filter(pl.col("call_put") == "P").select(
                "strike", pl.col("mid").alias("put_mid")
            ),
            on="strike",
            how="inner",
        )
        .with_columns((pl.col("call_mid") - pl.col("put_mid")).abs().alias("gap"))
        .sort("gap")
    )
    if pairs.height < _MIN_PARITY_PAIRS:
        raise ValueError("no strike has a two-sided call and put; cannot locate the forward")
    best = pairs.row(0, named=True)
    return float(best["strike"]) + math.exp(r * t_years) * (
        float(best["call_mid"]) - float(best["put_mid"])
    )


def _wing_quotes(
    frame: pl.DataFrame,
    k0: float,
    call_side: bool,
    stop_after: int,
) -> pl.DataFrame:
    """OTM strikes on one wing under the zero-bid truncation rule."""
    side = frame.filter(
        (pl.col("call_put") == ("C" if call_side else "P"))
        & ((pl.col("strike") > k0) if call_side else (pl.col("strike") < k0))
    ).sort("strike", descending=not call_side)

    included: list[tuple[float, float]] = []
    zero_run = 0
    for row in side.iter_rows(named=True):
        bid, ask = float(row["bid"] or 0.0), float(row["ask"] or 0.0)
        if bid <= 0.0:
            zero_run += 1
            if zero_run >= stop_after:
                break
            continue
        zero_run = 0
        if ask <= bid:
            continue  # crossed quote: unusable but does not truncate the wing
        included.append((float(row["strike"]), (bid + ask) / 2.0))
    return pl.DataFrame(
        {"strike": [s for s, _ in included], "q": [q for _, q in included]},
        schema={"strike": pl.Float64(), "q": pl.Float64()},
    )


def single_expiry_variance(
    frame: pl.DataFrame,
    *,
    t_years: float,
    r: float,
    cfg: ImpliedConfig,
    fallback_forward: float | None = None,
) -> ExpiryVariance:
    """The white-paper variance for one expiry slice (strike, call_put, bid, ask).

    ``fallback_forward`` handles thin ETF chains where no strike is two-sided
    on both call and put (observed on near-term FXE): the caller supplies the
    carry forward S·e^{(r-q)T} and the parity forward is used whenever it
    exists.
    """
    if t_years <= 0.0:
        raise ValueError(f"t_years must be positive, got {t_years}")
    try:
        forward = _forward_from_parity(frame, t_years, r)
    except ValueError:
        if fallback_forward is None:
            raise
        logger.info("no parity pair; using carry forward %.4f", fallback_forward)
        forward = fallback_forward

    # K0: the largest strike at or below F carrying at least one two-sided
    # quote — on thin ETF chains the literal white-paper K0 can be quoteless,
    # and skipping the ATM strike would bias the strip down far more than
    # stepping K0 to the nearest quoted strike does
    quoted_below = _two_sided(frame).filter(pl.col("strike") <= forward)["strike"]
    if quoted_below.len() == 0:
        raise ValueError("no quoted strike at or below the forward; chain too sparse")
    k0 = float(quoted_below.max())  # type: ignore[arg-type]

    puts = _wing_quotes(frame, k0, call_side=False, stop_after=cfg.zero_bid_consecutive_stop)
    calls = _wing_quotes(frame, k0, call_side=True, stop_after=cfg.zero_bid_consecutive_stop)

    # at K0 the quote is the average of the call and put mids (those present)
    at_k0 = _two_sided(frame.filter(pl.col("strike") == k0))
    k0_q = float(at_k0["mid"].mean())  # type: ignore[arg-type]

    table = (
        pl.concat([puts, pl.DataFrame({"strike": [k0], "q": [k0_q]}), calls])
        .sort("strike")
        .unique(subset="strike", keep="first", maintain_order=True)
    )
    strikes = table["strike"].to_list()
    quotes = table["q"].to_list()
    n = len(strikes)
    if n < _MIN_TERM_EXPIRIES:
        raise ValueError("fewer than two usable strikes after truncation")

    total = 0.0
    growth = math.exp(r * t_years)
    for i in range(n):
        if i == 0:
            dk = strikes[1] - strikes[0]
        elif i == n - 1:
            dk = strikes[-1] - strikes[-2]
        else:
            dk = (strikes[i + 1] - strikes[i - 1]) / 2.0
        total += dk / strikes[i] ** 2 * growth * quotes[i]

    sigma_sq = (2.0 / t_years) * total - (1.0 / t_years) * (forward / k0 - 1.0) ** 2
    return ExpiryVariance(sigma_sq=sigma_sq, forward=forward, k0=k0, n_options=n, t_years=t_years)


def select_term_expiries(
    minutes_by_expiry: dict[date, float],
    cfg: ImpliedConfig,
) -> tuple[date, date]:
    """(near, next) expiries around the constant-maturity target.

    Prefers the pair straddling the target; falls back to the two closest on
    one side (linear extrapolation, CBOE's own fallback) and logs the fact.
    """
    day_minutes = 24.0 * 60.0
    eligible = {
        expiry: minutes
        for expiry, minutes in minutes_by_expiry.items()
        if cfg.min_days_to_expiry * day_minutes <= minutes <= cfg.max_days_to_expiry * day_minutes
    }
    if len(eligible) < _MIN_TERM_EXPIRIES:
        raise ValueError(
            f"need at least two expiries within [{cfg.min_days_to_expiry}, "
            f"{cfg.max_days_to_expiry}] days, found {len(eligible)}"
        )
    target = cfg.target_days * day_minutes
    at_or_below = [e for e, m in eligible.items() if m <= target]
    above = [e for e, m in eligible.items() if m > target]
    if at_or_below and above:
        return max(at_or_below, key=lambda e: eligible[e]), min(above, key=lambda e: eligible[e])
    ordered = sorted(eligible, key=lambda e: abs(eligible[e] - target))
    near, nxt = sorted(ordered[:2], key=lambda e: eligible[e])
    logger.info(
        "no expiry pair straddles %d days; extrapolating from %s, %s", cfg.target_days, near, nxt
    )
    return near, nxt


def interpolate_constant_maturity(
    var_near: float,
    minutes_near: float,
    var_next: float,
    minutes_next: float,
    target_days: int,
) -> float:
    """White-paper minutes-weighted interpolation to the target maturity.

    Returns the *annualised* constant-maturity variance.
    """
    if minutes_near >= minutes_next:
        raise ValueError("near-term expiry must precede next-term expiry")
    n30 = target_days * 24.0 * 60.0
    t1 = minutes_near / MINUTES_PER_YEAR
    t2 = minutes_next / MINUTES_PER_YEAR
    w = (minutes_next - n30) / (minutes_next - minutes_near)
    total_30d = t1 * var_near * w + t2 * var_next * (1.0 - w)
    return total_30d * MINUTES_PER_YEAR / n30


def index_level(annualized_variance: float) -> float:
    """VIX-style quotation: 100 × annualised volatility."""
    if annualized_variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {annualized_variance}")
    return 100.0 * math.sqrt(annualized_variance)
