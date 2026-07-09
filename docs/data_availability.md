# Data availability — Phase 0 reconnaissance

**Date of reconnaissance:** 2026-07-08/09.
**Verdict: Index track.** FXE (and every other currency ETF) is entirely absent from the
DoltHub options database. Details and evidence below.

---

## 1. DoltHub `post-no-preference/options` — historical option chains

Queried via the DoltHub SQL API (`https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master`),
which avoids the multi-GB clone. Branch is `master`, not `main`.

**Database facts:**

- Tables: `option_chain`, `volatility_history`.
- `option_chain` columns: `date, act_symbol, expiration, strike, call_put, bid, ask, vol (IV), delta, gamma, theta, vega, rho`.
  **No volume, no open interest, no last-trade price.**
- Coverage: 2019-02-09 → 2026-07-07 (actively updated).
- Snapshot cadence is **irregular**, not strictly daily: e.g. data exists on Mon 2024-06-03
  and Mon 2024-06-10 but not Wed 2024-06-12.
- Chains are **filtered**, not full: SPY on 2024-06-10 has only 98 rows across 3
  expirations (10/17/22 strikes each). The real SPY chain has thousands of contracts.
  The dataset is a stock-screener extract (near-the-money, few tenors), not a
  surface-quality panel.

### Required coverage table

| Symbol | First date | Last date | Trading days | Mean strikes/day | Two-sided quote fraction |
|--------|-----------|-----------|--------------|------------------|--------------------------|
| FXE    | — absent  | — absent  | 0            | —                | —                        |
| FXY    | — absent  | — absent  | 0            | —                | —                        |
| FXB    | — absent  | — absent  | 0            | —                | —                        |
| UUP    | — absent  | — absent  | 0            | —                | —                        |

**Evidence of absence** (not merely sparseness):

1. `volatility_history` full scans: `COUNT(*) = 0` for FXE, FXY, FXB, UUP.
2. 120 quarterly `option_chain` probes (30 dates × 4 symbols, 2019–2026, PK-prefix
   lookups): 0 rows everywhere.
3. `SELECT DISTINCT act_symbol ... WHERE date='2024-06-10' AND act_symbol LIKE 'FX%'`
   on a known-good date: empty.
4. Control symbols on the same date: SPY = 98 rows (present), QQQ = 0, GLD = 0, USO = 0.
   The universe excludes ETFs generally, not just currency ETFs.

**Decision-gate consequence:** the *Full Surface* track is impossible from this source.
Even if the symbols had been present, the filtered strike grid (no OTM wings) and the
missing volume/OI fields would have made MFIV/BKM work marginal. **Do not clone this
database.**

Paid/registered alternatives for historical FXE chains, deliberately not pursued in
Phase 0 (would change project cost/licensing): CBOE DataShop, ORATS, OptionMetrics
(academic access). Noted for the limitations section.

## 2. Implied volatility index — EVZ

| Source | Series | First | Last | Obs |
|--------|--------|-------|------|-----|
| FRED   | `EVZCLS` | 2007-11-01 | 2025-03-11 | 4,529 |
| CBOE   | `EVZ_History.csv` (cdn.cboe.com) | 2009-09-18 | 2025-03-11 | 3,891 |

**EVZ is dead, not just the FRED mirror.** CBOE's own history file also stops at
2025-03-11 — the index was decommissioned. There is no EVZ after that date and there
will not be. The long-horizon backtest sample is therefore **hard-capped at
2007-11-01 → 2025-03-11** on the implied side. FRED's copy is the longer one; use it,
cross-checked against CBOE's file on the overlap.

Corollary: our own MFIV computed from forward-collected FXE chains is not merely a
validation exercise — it is the *only* continuation of this series that will exist.

`VIXCLS` (control variable): 1990-01-02 → current, no issues.

## 3. Spot / realised volatility — Dukascopy EURUSD ticks

Verified by direct download. Data has **bid and ask separately**, no volume (correctly
ignored).

| Sample day | Ticks | Median spread | p95 spread | Crossed quotes | Max gap |
|------------|-------|---------------|------------|----------------|---------|
| 2020-03-09 (COVID stress) | 293,707 | 0.5 pips | 1.1 pips | 0 | 114.6 s |
| 2007-06-12 (10:00 hour only, raw `.bi5`) | 452/hour | 1.0 pips | — | 0 | — |

- History confirmed back to at least mid-2007 (the 2007→2025 window we need).
- Tooling: `dukascopy-node` works but **must be throttled**: pin
  `-bs 5 -bp 500 -r 3` (batch size 5, 500 ms pause, 3 retries). Default settings hit
  rate limiting ("fetch failed").
- `dukascopy-node` errored on a 2007 date ("Unknown error") while the raw feed has the
  data; fallback ingester = direct fetch of
  `datafeed.dukascopy.com/datafeed/EURUSD/{yyyy}/{MM-1}/{dd}/{HH}h_ticks.bi5`
  (month is zero-indexed) + LZMA decompress + big-endian `>IIIff` structs
  (ms-offset, ask×1e5, bid×1e5, askVol, bidVol). Verified working. Phase 1 decides
  which path is primary; both are documented.
- Volume estimate: ~8.3 MB/day CSV in stress → order 25–40 GB raw for 2007–2025;
  ingest to partitioned parquet on a scratch disk.

## 4. Rates

| Series | Source | Coverage | Role |
|--------|--------|----------|------|
| `DFF` (fed funds effective) | FRED | 1954 → current | USD short rate, discounting |
| `EONIARATE` | FRED | 1999-01-04 → 2021-12-29 (discontinued) | EUR overnight, pre-2022 |
| €STR (`EST.B.EU000A2X2A25.WT`) | ECB Data Portal API | 2019-10-01 → current | EUR overnight, 2019+ |

EUR overnight rate must be **stitched**: EONIA until 2019-09-30, €STR thereafter
(EONIA was redefined as €STR + 8.5 bp from Oct 2019, so the splice is clean; the 8.5 bp
methodological break is documented in `conventions.md`). ECB CSV API verified:
`data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata`.

## 5. Forward-collected FXE chains — the scraper source

CBOE delayed quotes JSON, verified live on 2026-07-08:
`https://cdn.cboe.com/api/global/delayed_quotes/options/FXE.json`

- FXE still listed and optionable; spot 105.31 at recon time.
- 244 contracts, 4 expiries (Jul/Aug/Sep/Dec 2026), ~29–32 strikes per expiry.
- Fields: bid, ask, bid_size, ask_size, IV, all greeks, **open interest, volume**,
  last trade. Far richer than the DoltHub schema.
- Two-sided quote fraction: 57.4% (wings are zero-bid, as expected; CBOE's own
  zero-bid truncation rules handle this in MFIV).

**Design adaptation forced by EVZ's death:** published EVZ ended 2025-03-11, so a
forward-collected FXE panel has *no overlap* with it — the Phase 3 acceptance test
"replicate published EVZ" cannot run on forward data. Fix: the scraper also collects
`_SPX` chains daily (same endpoint pattern, verified to exist) and we validate the MFIV
implementation by **replicating published VIX** on the forward window instead. Same
methodology, live index to check against, then applied to FXE. Optionally also scrape
FXY/FXB/UUP (cheap) for cross-sectional colour later.

## 6. FXE mechanics (early-exercise premium — required task, §1.5)

CRR binomial (800 steps), American vs European, FXE distributions approximated as a
continuous foreign-currency yield `q` (EUR deposit rate). Premium = (Am − Eu)/Eu.
Grid: K/S ∈ {0.97…1.03}, σ ∈ {6%, 9%, 12%}, T ∈ {30d, 60d}, four rate regimes.

| Regime (r_USD, q_EUR) | 30d ATM | 60d ATM | 30d 5%-OTM | Worst corner (ITM, low vol) |
|---|---|---|---|---|
| 2015–21 (0.25%, −0.50%) | put 0.34% | put 0.53% | put 0.17% | put 1.5% |
| Current (3.60%, 2.20%) | put 0.82% | put 1.32% | put 0.38% | put 4.0% |
| 2023 peak (5.50%, 3.90%) | put 1.00% | put 1.65% | put 0.45% | put 5.0% |
| Inverted (2.00%, 4.00%) | call 1.27% | call 2.06% | call 0.60% | call 6.6% |

The premium sits on the side whose exercise captures the rate differential (puts when
r > q, calls when q > r), grows with |r − q| and tenor, and concentrates ITM.

**Decision (measured, per the brief):** the blanket European approximation does *not*
clear a 1% bar uniformly — 30d ATM is 0.3–1.3% depending on regime, 60d up to ~2%.
Therefore: (a) IV extraction from FXE quotes uses **binomial de-Americanization**
(`implied/american.py`, already planned); (b) MFIV uses OTM quotes only, where the
worst case is ≈0.6–1.2%, and the residual bias is reported, not ignored; (c) the
European pricer remains the analytic workhorse everywhere the simulation defines the
ground truth. The continuous-yield approximation of FXE's *discrete monthly*
distributions is a second-order refinement; revisit in Phase 3 if the reconciliation
residual demands it.
