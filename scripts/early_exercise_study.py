"""The FXE early-exercise premium study (brief §1.5), on library code.

Reproduces the Phase 0 measurement with `fxvrp.implied.american.crr_price` and
writes docs/reports/early_exercise.md. The conclusion drives the standing
decision: listed FXE IVs are extracted by de-Americanization, MFIV uses OTM
quotes only, and the European model remains the analytic core.
"""

from __future__ import annotations

from fxvrp.config import load_config
from fxvrp.implied import crr_price
from fxvrp.log import get_logger

logger = get_logger("scripts.early_exercise_study")

SPOT = 100.0
REGIMES = {
    "2015-21 (r=0.25%, q=-0.50%)": (0.0025, -0.0050),
    "current (r=3.60%, q=2.20%)": (0.0360, 0.0220),
    "2023 peak (r=5.50%, q=3.90%)": (0.0550, 0.0390),
    "inverted (r=2.00%, q=4.00%)": (0.0200, 0.0400),
}
TENORS = {"30d": 30.0 / 365.0, "60d": 60.0 / 365.0}
VOLS = (0.06, 0.09, 0.12)
MONEYNESS = (0.97, 0.99, 1.00, 1.01, 1.03)
MIN_PRICE = 0.01  # ignore near-worthless corners


def premium(r: float, q: float, t: float, sigma: float, k: float, call: bool, steps: int) -> float:
    eu = crr_price(
        s=SPOT, k=k, t=t, r=r, q=q, sigma=sigma, n_steps=steps, call=call, american=False
    )
    if eu < MIN_PRICE:
        return 0.0
    am = crr_price(s=SPOT, k=k, t=t, r=r, q=q, sigma=sigma, n_steps=steps, call=call, american=True)
    return (am - eu) / eu


def main() -> None:
    config = load_config()
    steps = config.implied.binomial_steps
    lines = [
        "# FXE early-exercise premium (American - European) / European\n\n",
        "CRR binomial, continuous EUR-yield approximation of FXE distributions; ",
        f"{steps} steps. Grid: K/S in {MONEYNESS}, vol in {VOLS}.\n\n",
        "| regime | tenor | ATM put | ATM call | worst corner |\n|---|---|---|---|---|\n",
    ]
    worst_overall = 0.0
    for name, (r, q) in REGIMES.items():
        for tenor_name, t in TENORS.items():
            atm_put = premium(r, q, t, VOLS[1], SPOT, call=False, steps=steps)
            atm_call = premium(r, q, t, VOLS[1], SPOT, call=True, steps=steps)
            corner = max(
                premium(r, q, t, sigma, SPOT * m, call=cp, steps=steps)
                for sigma in VOLS
                for m in MONEYNESS
                for cp in (True, False)
            )
            worst_overall = max(worst_overall, corner)
            lines.append(
                f"| {name} | {tenor_name} | {atm_put:.2%} | {atm_call:.2%} | {corner:.2%} |\n"
            )
    lines.append(
        f"\nWorst case across the grid: **{worst_overall:.2%}** of option value.\n\n"
        "Decision (standing): premiums of this size cannot be waved away, so FXE "
        "IV extraction de-Americanizes against the binomial; OTM-only MFIV keeps "
        "the residual bias near or below ~1%.\n"
    )
    out = config.paths.reports_dir / "early_exercise.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    logger.info("wrote %s (worst case %.2f%%)", out, 100 * worst_overall)


if __name__ == "__main__":
    main()
