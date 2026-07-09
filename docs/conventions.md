# Conventions memo — fixed for the life of the project

Every module obeys these. Changing any of them requires an ADR in `docs/decisions/`.

## Time, day count, annualisation

1. **All timestamps stored in UTC.** Dukascopy is natively UTC. US market times are
   converted via the `America/New_York` tz database (never a fixed offset — DST).
2. **Day count: ACT/365 (calendar)** for option time-to-expiry and variance
   annualisation, matching the VIX/EVZ methodology. Where we replicate the index
   (MFIV, 30-day interpolation), time is measured in **minutes**, per the CBOE
   white paper.
3. **The VRP window is 30 calendar days**, not 21 trading days. Both legs of the VRP
   use the identical window and the identical annualisation factor (365/30 on
   30-day-window variance). The 252-day convention appears nowhere in the VRP
   definition; it may appear only inside auxiliary diagnostics and must never cross
   into the implied-vs-realised comparison.

## Alignment of the two legs

4. **Observation timestamp:** the implied leg (EVZ close) is stamped at the US index
   close, 16:15 ET. The realised window for date *t* runs from **16:15 ET on t to
   16:15 ET on t+30 calendar days**, computed from tick data. No same-day information
   later than 16:15 ET may enter a signal dated *t*.
5. **Weekend and holiday variance is included on both legs.** EVZ prices variance over
   calendar time (weekends included in the minute count); therefore realised variance
   includes the FX weekend gap: the squared log return from the last Friday tick to the
   first Sunday-reopen tick is part of the window's RV. A weekend-excluded variant is
   computed once as a sensitivity table for the paper, never used in signals.
6. **FX day boundary** for daily RV bucketing: 17:00 ET (the FX value-date roll),
   i.e. a "day" is 17:00 ET → 17:00 ET. Days with fewer than a configured minimum of
   ticks (holidays, outages) are flagged, logged, and reported — not silently padded.

## Realised-variance measurement

7. **Prices are log mid-quotes**, mid = (bid+ask)/2. Bid/ask bounce must not enter RV;
   the spread enters the *cost model* instead, as a first-class series.
8. **Baseline estimator: 5-minute calendar-time RV**, subsample-averaged; TSRV and the
   realised kernel are the noise-robust cross-checks. Sampling-frequency choices live
   in `configs/default.yaml`, and the volatility signature plot justifies the baseline.
9. Returns spanning the weekend gap belong to the window per rule 5 but are excluded
   from *intraday* estimator inputs that assume equispaced diffusion sampling (BPV,
   TQ, jump tests) — the gap return is handled as a separate additive term. Rationale:
   bipower's jump-robustness logic does not survive a 48-hour hole; document in the
   paper.

## Rates, carry, discounting

10. **USD short rate: fed funds effective (DFF/EFFR)**, quoted ACT/360 simple;
    converted to continuously-compounded ACT/365 before use. For tenors ≤ 60 days the
    flat-overnight proxy for the OIS term rate is accepted; the approximation error
    (a few bp of rate ≈ negligible option-price effect) is stated in the paper.
11. **EUR short rate: EONIA until 2019-09-30, €STR from 2019-10-01.** The 8.5 bp
    redefinition spread is left as-is (no back-adjustment); the splice date and the
    break are disclosed. Carry = r_USD − r_EUR from these series.
12. **Discounting for option calculations uses the USD leg only** (options are
    USD-denominated ETF options). The EUR rate enters as the distribution yield of FXE.

## Options data and implied-side rules

13. **FXE distributions are modelled as a continuous yield q = EUR overnight rate.**
    Revisit (ADR required) if Phase 3 reconciliation shows a residual attributable to
    discrete-distribution timing.
14. **IV extraction from FXE quotes de-Americanizes** via CRR binomial (Phase 0
    measurement: ATM early-exercise premium 0.3–2.1% of option value depending on
    regime and tenor — too big to wave away, small enough that the European model
    remains the analytic core).
15. **MFIV uses OTM options only**, mid-quotes, CBOE zero-bid and
    two-consecutive-zero-bid truncation rules, forward via put-call parity at the
    minimum-|C−P| strike. No improvised strike-selection rules.
16. **Quote filters** (chains): drop bid ≤ 0, crossed (ask ≤ bid), and stale rows —
    every filter logs rows in / rows out / reason.
17. **The scraper snapshot time** for the forward-collected panel is fixed at one
    time-of-day and recorded per row; mixed-time snapshots are never merged silently.
    *Amended (Phase 3):* the primary snapshot runs mid-morning US time (21:45
    Bangkok ≈ 09:45/10:45 ET) because the post-close delayed-quote state for FXE
    was observed frozen with zero-bid OTM wings (2026-07-08), which is unusable
    for MFIV; a 05:00 Bangkok fallback run still catches any missed day. The
    per-row `quote_time` is authoritative.

## Signals and hygiene

18. **Ex-ante vs ex-post:** any tradeable signal uses only information timestamped
    ≤ the decision time (rule 4). The backtest engine carries a lookahead guard that
    raises on violation; the guard itself is under test.
19. **No fabricated data:** no interpolation/forward-fill of missing market data
    without an explicit, logged, ADR-documented decision.
20. **All magic numbers live in `configs/default.yaml`.**
