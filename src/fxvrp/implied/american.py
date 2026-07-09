"""American option pricing (CRR binomial) and de-Americanization.

Cox, Ross & Rubinstein (1979, J. Financial Economics 7): recombining tree with
u = e^{σ√dt}, d = 1/u, risk-neutral probability p = (e^{(r-q)dt} - d)/(u - d).
The continuous yield q approximates FXE's discrete monthly distributions
(Phase 0 decision, `docs/data_availability.md` §6): the early-exercise premium
measured there (ATM 0.3-2.1% of value depending on regime and tenor) is too
large to ignore in IV extraction, so listed FXE quotes are inverted against the
*American* price ("de-Americanization") before any European machinery runs.

Convergence: 500 steps prices our tenor/moneyness grid within 0.02% of the
800-step Phase 0 reference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from fxvrp.implied.black_scholes import IV_MAX, IV_MIN

_PRICE_EPS = 1e-12


def crr_price(
    *,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    call: bool,
    american: bool,
) -> float:
    """CRR binomial price, European or American."""
    if s <= 0.0 or k <= 0.0:
        raise ValueError(f"spot and strike must be positive, got s={s}, k={k}")
    if t < 0.0 or sigma < 0.0:
        raise ValueError("tenor and sigma must be non-negative")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if t == 0.0:
        return max(s - k, 0.0) if call else max(k - s, 0.0)
    if sigma == 0.0:
        # degenerate deterministic tree; fall back to discounted intrinsic
        intrinsic = s * math.exp(-q * t) - k * math.exp(-r * t)
        european = max(intrinsic, 0.0) if call else max(-intrinsic, 0.0)
        if not american:
            return european
        return max(european, (s - k) if call else (k - s), 0.0)

    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside (0,1); increase n_steps or check r, q, sigma"
        )
    disc = math.exp(-r * dt)

    j = np.arange(n_steps + 1)
    terminal = s * u ** (n_steps - j) * d**j
    values = np.maximum((terminal - k) if call else (k - terminal), 0.0)
    for step in range(n_steps - 1, -1, -1):
        j = np.arange(step + 1)
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            spots = s * u ** (step - j) * d**j
            exercise = (spots - k) if call else (k - spots)
            values = np.maximum(values, exercise)
    return float(values[0])


def de_americanize(
    *,
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    call: bool,
    n_steps: int,
) -> float:
    """Implied volatility from an *American* option quote via bracketed Brent.

    This is the volatility that, fed to a European model, is free of the
    early-exercise premium embedded in listed FXE quotes.
    """
    if t <= 0.0:
        raise ValueError("cannot invert implied vol at expiry (t=0)")
    intrinsic = max(s - k, 0.0) if call else max(k - s, 0.0)
    lower = crr_price(s=s, k=k, t=t, r=r, q=q, sigma=0.0, n_steps=1, call=call, american=True)
    if price < max(intrinsic, lower) - _PRICE_EPS:
        raise ValueError(f"price {price} below American lower bound {max(intrinsic, lower):.6g}")
    if price <= lower + _PRICE_EPS:
        return IV_MIN

    def objective(sigma: float) -> float:
        return (
            crr_price(
                s=s, k=k, t=t, r=r, q=q, sigma=sigma, n_steps=n_steps, call=call, american=True
            )
            - price
        )

    # CRR requires p in (0,1), i.e. sigma > |r-q| sqrt(dt); lift the bracket floor
    sigma_floor = max(IV_MIN, abs(r - q) * math.sqrt(t / n_steps) * (1.0 + 1e-6))
    if objective(sigma_floor) > 0.0:
        return sigma_floor  # quote at or below the minimum-vol price
    if objective(IV_MAX) < 0.0:
        raise ValueError(f"price {price} implies volatility above {IV_MAX:.0%}")
    return float(optimize.brentq(objective, sigma_floor, IV_MAX, xtol=1e-8, rtol=1e-10))
