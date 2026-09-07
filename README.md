# The FX variance risk premium (on-going)

**Is the FX variance risk premium compensation for jump/crash risk or for diffusive
risk, and, separately, can it actually be harvested once discrete delta-hedging error
and realistic transaction costs are accounted for?** Those are two different questions.
The P&L of a delta-hedged short straddle is not the raw implied-minus-realised variance
gap, but the **gamma-weighted** gap

```
Π ≈ ∫₀ᵀ ½ Γᵤ Sᵤ² (σ²_implied − σ²_realised,u) du ,
```

## Status

| Phase | Deliverable | State |
|-------|-------------|-------|
| 0 | Data reconnaissance, track decision, conventions | done, see `docs/` |
| 1 | Foundation: simulators, tick/rates/chain data layer, daily chain scraper | done (`v0.1.0`) |
| 2 | Realised variance estimators, validated on simulated ground truth | done (`v0.2.0`) |
| 3 | Implied variance: MFIV, VIX replication, de-Americanization | done (`v0.3.0`) |
| 4 | The VRP series: HAR forecasts, ex-ante vs ex-post | **current** |
| 5 | Strategy, P&L attribution, the hedging-frequency study | - |
