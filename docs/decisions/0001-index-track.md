# 0001 — Index track: EVZ as the historical implied series

**Status:** accepted (Phase 0 gate).

## Context

The planned free historical FXE chain source (DoltHub `post-no-preference/options`)
contains no currency ETFs at all — FXE, FXY, FXB and UUP have zero rows over its
entire 2019–2026 history, the universe excludes ETFs generally, and even present
symbols carry only screener-filtered chains without OTM wings, volume or open
interest. Independently, CBOE decommissioned the EVZ index on 2025-03-11.
Evidence: `docs/data_availability.md` §1–2.

## Options considered

1. **Full Surface track** — impossible without paid data (CBOE DataShop, ORATS,
   OptionMetrics), which changes cost and licensing.
2. **Index track** — EVZ (FRED `EVZCLS`, 2007-11-01 → 2025-03-11) as the implied
   series for the long-horizon backtest, plus a forward-collected FXE surface panel
   accruing daily from Phase 1 onwards.

## Decision

Index track. The backtest sample is hard-capped at 2025-03-11 on the implied side,
stated in the paper rather than worked around.

## Consequences

- Surface machinery (MFIV, BKM, SVI) is demonstrated and validated on the forward
  panel, not backtested on history.
- The forward panel has no overlap with published EVZ, which forces ADR 0002.
- Our MFIV series is the only continuation of a discontinued index — a feature of
  the paper, not an apology.
