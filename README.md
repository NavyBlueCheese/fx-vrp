# The FX variance risk premium (on-going)

**Is the FX variance risk premium compensation for jump/crash risk or for diffusive
risk, and, separately, can it actually be harvested once discrete delta-hedging error
and realistic transaction costs are accounted for?** Those are two different questions.
The P&L of a delta-hedged short straddle is not the raw implied-minus-realised variance
gap, but the **gamma-weighted** gap

```
Π ≈ ∫₀ᵀ ½ Γᵤ Sᵤ² (σ²_implied − σ²_realised,u) du ,
```

so you can be exactly right about 30-day variance and still lose money. This project
measures both objects on EUR/USD, which is the implied side from the CBOE EVZ index and FXE
option chains, the realised side from Dukascopy tick data with real bid/ask spreads, 
and decomposes every trade's P&L into variance gap, discrete hedging error, option
spread cost, spot spread cost, and carry, with the decomposition reconciled to the
total as a unit test.

*Headline figure (hedging-error std ~ n^(-1/2) vs. linear transaction cost, and the
empirically optimal rehedge frequency) lands here at Phase 5.*

## Status

| Phase | Deliverable | State |
|-------|-------------|-------|
| 0 | Data reconnaissance, track decision, conventions | done, see `docs/` |
| 1 | Foundation: simulators, tick/rates/chain data layer, daily chain scraper | **current** |
| 2 | Realised variance estimators, validated on simulated ground truth | - |
| 3 | Implied variance: MFIV, VIX replication, de-Americanization | - |
| 4 | The VRP series: HAR forecasts, ex-ante vs ex-post | - |
| 5 | Strategy, P&L attribution, the hedging-frequency study | - |

**Track note:** the free historical FXE chain source evaluated in Phase 0 turned out
to contain no currency ETFs at all, and CBOE decommissioned the EVZ index on
2025-03-11. The long-horizon backtest therefore uses EVZ (2007-11 → 2025-03) as the
implied series, while a daily scraper collects the live FXE surface going forward, which is
the only continuation of the discontinued index. Details: `docs/data_availability.md`.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). `data/` is never committed (everything is
rebuilt from public sources)

```sh
uv sync                 # environment (Python 3.11+, locked)
make test               # unit tests + property tests against simulated ground truth
make data-sample        # bounded data pull: FRED + ECB + one month of ticks + chains
make data               # full 2007-2025 tick ingestion (long; resumable)
make quality-report     # data-quality report -> docs/reports/
```

Without `make` on Windows, each target is a one-liner documented in the `Makefile`.

## Layout

```
configs/default.yaml    every numeric assumption in the project, in one place
docs/                   data availability, fixed conventions, ADRs, quality reports
src/fxvrp/simulate/     GBM / Heston / Merton / microstructure ground-truth worlds
src/fxvrp/data/         Dukascopy ticks, FRED & ECB rates, CBOE chain scraper
tests/                  every estimator validated against a world with a known answer
scripts/                one entry point per pipeline step / paper figure
```

## Data sources

Dukascopy EURUSD ticks (bid & ask, 2007 ->), FRED (`EVZCLS`, `VIXCLS`, `DFF`,
`EONIARATE`), ECB Data Portal (€STR), CBOE delayed quotes (FXE, SPX chains).
All free; access details and quality findings in `docs/data_availability.md`.
