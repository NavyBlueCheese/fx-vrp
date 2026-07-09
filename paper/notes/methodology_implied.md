# Methodology notes — implied variance (drafted during Phase 3)

Claims pinned by `tests/test_implied_*.py`.

## Why the 1/K² strip prices variance

For a diffusion, Itô on log S gives
d(log S) = dS/S − ½σ²dt  ⇒  ∫₀ᵀ σ²dt = 2[∫₀ᵀ dS/S − log(S_T/S₀)].
The payoff −log(S_T/F) (the "log contract") is replicated statically by
∫₀^F P(K)/K² dK + ∫_F^∞ C(K)/K² dK (Carr–Madan expansion: any twice-differentiable
payoff is a strip of OTM options weighted by its second derivative; here
f(S) = −log S has f''(K) = 1/K²). Hence risk-neutral expected variance is the
discounted OTM strip — no model for σ needed, which is what "model-free" means.
The −(1/T)(F/K₀−1)² term corrects for using K₀ instead of F as the put/call split.
Under Black-Scholes with constant σ the strip returns exactly σ² — the test.

## Discretisation and truncation

ΔKᵢ/Kᵢ² approximates the integral on the listed grid; zero-bid truncation cuts the
tails. Both errors are *downward* (lost wing mass), which is why the truncation test
asserts the truncated estimate is below the full one, and the sensitivity test shows
±6σ wings contribute nothing. On real chains, wing truncation is the binding error
and is reported per day (strike counts in the reconciliation report).

## The forward and K₀

Parity at the minimising strike: C − P = e^{-rT}(F − K), so
F = K* + e^{rT}(C(K*) − P(K*)) with K* = argmin|C−P| — the strike where parity is
least distorted by spread. K₀ = largest strike ≤ F splits the wings.

## De-Americanization

Listed FXE options are American. The measured early-exercise premium
(docs/reports/early_exercise.md) reaches ~1% ATM / ~5-7% ITM in high-|r−q| regimes,
sitting on the side whose exercise captures the rate differential (puts when r>q).
So quoted prices are inverted against the CRR American price, not the European
formula; the recovered σ is then safe for European machinery. Verified by
round-trip (σ → American price → σ) and by showing naive European inversion
overstates vol on exactly the predicted side.

## Constant-maturity interpolation

Total variance Tσ² is the additive object; the white paper interpolates it linearly
in *minutes* to the 30-day point and re-annualises. AM/PM settlement matters at this
precision: standard third-Friday SPX settles 09:30 ET, weeklies and ETF options
16:00 ET.

## Standing decisions taken autonomously (delegated at the Phase 2 gate)

1. **VIX replication replaces EVZ replication** as the MFIV acceptance test
   (ADR 0002) — EVZ is dead; VIX is live, same methodology, harder test.
2. **Discounting**: DFF converted 365·ln(1+R/360); flat overnight proxy for ≤60d
   tenors, error ≪ the quote spread.
3. **Multi-root expiries** (SPX vs SPXW on one date): the root with more two-sided
   quotes wins the expiry; mixed-root slices are never blended.
4. **Kernel/TSRV citation checks** remain open TODO(verify) items; behaviour is
   test-pinned, and the paper will cite the estimators by construction, not by
   equation number, until verified against the primary sources.
