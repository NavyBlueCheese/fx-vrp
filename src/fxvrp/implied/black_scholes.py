"""Black-Scholes-Merton pricing, greeks, and implied-volatility inversion.

Merton (1973) dividend-yield form: for spot S, strike K, tenor T (years,
ACT/365), continuously-compounded rates r (domestic) and q (dividend /
foreign-currency yield),

    d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ√T),   d2 = d1 - σ√T,
    C = S e^{-qT} N(d1) - K e^{-rT} N(d2),
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1).

For FX, q is the foreign short rate (Garman & Kohlhagen 1983); for FXE, q is
the EUR deposit yield the trust distributes.

Inversion brackets the root with Brent's method on σ ∈ [IV_MIN, IV_MAX] after
checking static no-arbitrage bounds; vega-based Newton is not used because it
stalls for deep-OTM quotes where vega underflows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import optimize, stats

# solver brackets (algorithmic bounds, not market assumptions): vol in [1bp, 500%]
IV_MIN = 1e-4
IV_MAX = 5.0
_PRICE_EPS = 1e-12


def _validate(s: float, k: float, t: float) -> None:
    if s <= 0.0 or k <= 0.0:
        raise ValueError(f"spot and strike must be positive, got s={s}, k={k}")
    if t < 0.0:
        raise ValueError(f"tenor must be non-negative, got {t}")


def bs_price(
    *, s: float, k: float, t: float, r: float, q: float, sigma: float, call: bool
) -> float:
    """European option price; degenerates to discounted intrinsic at t=0 or σ=0."""
    _validate(s, k, t)
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if t == 0.0:
        return max(s - k, 0.0) if call else max(k - s, 0.0)

    forward_leg = s * math.exp(-q * t)
    strike_leg = k * math.exp(-r * t)
    if sigma == 0.0:
        intrinsic = forward_leg - strike_leg
        return max(intrinsic, 0.0) if call else max(-intrinsic, 0.0)

    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + sigma**2 / 2.0) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if call:
        return float(forward_leg * stats.norm.cdf(d1) - strike_leg * stats.norm.cdf(d2))
    return float(strike_leg * stats.norm.cdf(-d2) - forward_leg * stats.norm.cdf(-d1))


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # per unit of vol (not per vol point)
    theta: float  # per year
    rho: float  # per unit of rate


def bs_greeks(
    *, s: float, k: float, t: float, r: float, q: float, sigma: float, call: bool
) -> Greeks:
    """Analytic BSM greeks (Hull, *Options, Futures and Other Derivatives*, ch. 19)."""
    _validate(s, k, t)
    if t == 0.0 or sigma <= 0.0:
        raise ValueError("greeks need t > 0 and sigma > 0")

    sqrt_t = math.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (math.log(s / k) + (r - q + sigma**2 / 2.0) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    pdf_d1 = float(stats.norm.pdf(d1))

    if call:
        delta = disc_q * float(stats.norm.cdf(d1))
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            - r * k * disc_r * float(stats.norm.cdf(d2))
            + q * s * disc_q * float(stats.norm.cdf(d1))
        )
        rho = k * t * disc_r * float(stats.norm.cdf(d2))
    else:
        delta = -disc_q * float(stats.norm.cdf(-d1))
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            + r * k * disc_r * float(stats.norm.cdf(-d2))
            - q * s * disc_q * float(stats.norm.cdf(-d1))
        )
        rho = -k * t * disc_r * float(stats.norm.cdf(-d2))

    return Greeks(
        delta=delta,
        gamma=disc_q * pdf_d1 / (s * vol_sqrt_t),
        vega=s * disc_q * pdf_d1 * sqrt_t,
        theta=theta,
        rho=rho,
    )


def implied_vol(
    *, price: float, s: float, k: float, t: float, r: float, q: float, call: bool
) -> float:
    """Invert BSM for σ with a bracketed Brent solve.

    Raises ValueError when the quote violates static no-arbitrage bounds
    (below discounted intrinsic or above the spot/strike leg) — a bad quote is
    a data problem to surface, not a root to force.
    """
    _validate(s, k, t)
    if t == 0.0:
        raise ValueError("cannot invert implied vol at expiry (t=0)")

    lower = bs_price(s=s, k=k, t=t, r=r, q=q, sigma=0.0, call=call)
    upper = s * math.exp(-q * t) if call else k * math.exp(-r * t)
    if price < lower - _PRICE_EPS or price > upper + _PRICE_EPS:
        raise ValueError(f"price {price} outside no-arbitrage bounds [{lower:.6g}, {upper:.6g}]")
    if price <= lower + _PRICE_EPS:
        return IV_MIN

    def objective(sigma: float) -> float:
        return bs_price(s=s, k=k, t=t, r=r, q=q, sigma=sigma, call=call) - price

    if objective(IV_MAX) < 0.0:
        raise ValueError(f"price {price} implies volatility above {IV_MAX:.0%}")
    result = optimize.brentq(objective, IV_MIN, IV_MAX, xtol=1e-10, rtol=1e-12)
    return float(result)
