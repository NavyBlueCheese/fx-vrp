# 0003 — Tick ingestion: direct .bi5 fetch, not dukascopy-node

**Status:** accepted.

## Context

Phase 0 tested both access paths to Dukascopy EURUSD tick history. The
`dukascopy-node` CLI works when throttled (`-bs 5 -bp 500 -r 3`) but failed
opaquely on a 2007 date, adds a Node.js dependency to an otherwise-Python
pipeline, and its CSV output doubles storage before parquet conversion. The raw
feed (`datafeed.dukascopy.com/datafeed/{INST}/{yyyy}/{MM-1}/{dd}/{HH}h_ticks.bi5`,
zero-indexed months, LZMA-compressed `>IIIff` records) was decoded successfully in
pure Python, including for 2007.

## Decision

Ingest directly from the raw feed in `src/fxvrp/data/dukascopy.py`: fetch 24 hour
files per day with retries and throttling, decode, write one sorted parquet per
day. Resumable via skip-if-exists. The decoder is unit-tested against crafted
`.bi5` bodies; empty bodies (weekends/holidays) are valid and produce empty frames.

## Consequences

- One toolchain (Python), one storage format (parquet), fully typed and tested.
- We own the format assumption; a Dukascopy format change breaks loudly in the
  decoder's length check rather than silently in a third-party tool.
- Bid/ask volumes are discarded deliberately (meaningless in decentralised FX).
