# 0002 — CBOE delayed quotes as the chain source; VIX replication as MFIV validation

**Status:** accepted.

## Context

The Index track requires a forward-collected FXE chain panel starting on day one.
Phase 0 verified that CBOE's delayed-quotes endpoint
(`cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json`) serves the full
listed FXE chain with two-sided quotes, IV, greeks, open interest and volume, free
and keyless. Because published EVZ ended 2025-03-11, a forward panel can never be
validated against it.

## Options considered

1. Scrape a broker/Yahoo chain — weaker fields, unstable schemas, ToS friction.
2. Scrape CBOE delayed quotes — richest free source, and CBOE is the venue itself.
3. Buy historical chains to regain EVZ overlap — out of budget scope (ADR 0001).

## Decision

Scrape CBOE delayed quotes daily for `FXE`, `_SPX`, `FXY`, `FXB`, `UUP`. Validate
the MFIV implementation by **replicating published VIX from the `_SPX` snapshots**
(same methodology, live referee), then apply the identical code to FXE. Snapshot
timing: post-close scrape; the quote timestamp embedded in the payload (US/Eastern)
keys the snapshot date, and both fetch time and quote time are stored per row.

## Consequences

- Phase 3's acceptance test becomes "replicated VIX tracks published VIX", with the
  FXE MFIV series accepted on the strength of the shared implementation.
- Scraped snapshots are irreproducible history: `data/` stays gitignored per the
  engineering standards, so the panel must be included in the machine's normal
  backup routine (flagged at the Phase 1 gate).
- A missed scrape day is a permanent hole in the forward panel; the scheduled task
  uses catch-up-on-boot semantics to minimise misses, and gaps are reported, not
  filled.
