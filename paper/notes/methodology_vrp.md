# Methodology notes — the VRP series (drafted during Phase 4)

Claims pinned by `tests/test_har.py`, `tests/test_vrp_series.py`,
`tests/test_lookahead.py`, `tests/test_stylized_facts.py`.

## Two objects, never confused

VRP^{ex-post}_t = IV²_t − RV_{t,t+30} uses the future: it is the realisation the
paper *describes*. VRP^{ex-ante}_t = IV²_t − E_t[RV_{t,t+30}] replaces the future
with a forecast built from information at t: it is the only object a strategy may
touch. The code keeps them in separate columns and the backtest engine will read
only the ex-ante one; the lookahead guard turns contamination into a crash, and the
guard itself is tested with deliberately-fed future data.

## Leg alignment

Both legs are annualised 30-calendar-day variances. Implied: (EVZ/100)². Realised:
Σ daily RV over (t, t+30d] × 365/30, where daily RV comes from the 5-minute
previous-tick grid and **inter-day gap returns** (weekend, holiday) are added as
squared close-to-open log returns — the implied side charges for calendar time, so
the realised side must pay for it. A window missing ingested days is null, never
partially summed: an ingestion gap must not masquerade as low variance. The
16:15-ET EVZ print vs 17:00-ET window start (~0.1% of the window) is disclosed.

## HAR in logs, direct multi-horizon

log RV is far closer to Gaussian than RV (ABDL 2003), so the regression runs in
logs on the Corsi components (trailing 1/5/22-day mean RV, all information ≤ t) with
the target being the log *cumulative forward window* — a direct regression at the
30-day horizon rather than iterated one-step forecasts, which would compound
specification error 22 times. Level forecasts apply exp(σ̂²/2); the test shows the
uncorrected forecast is ~25%+ biased low when log-noise is large.

HAR-RV-CJ (ABD 2007) splits the daily regressor into continuous and jump parts by
the BNS test at 1%: C_t = BPV on jump days, RV otherwise; J_t = (RV−BPV)⁺ on jump
days; the jump regressor enters as log(1+J). Whether β_J < β_C (jumps forecast
less future variance than diffusion) is directly relevant to the paper's question.

## Walk-forward discipline

Expanding window, refit every 21 rows, and — the subtle requirement — the training
set at decision date t contains only rows whose own forward window has **closed**
by t (day ≤ t − 30d). Without that trim, the most recent training targets would
quietly contain realised variance from after t. Acceptance test: delete all data
after t and the forecast at t is bit-identical.

## Inference under overlap

Adjacent 30-day windows share 29 days, inducing AC(1) ≈ 0.97 in the ex-post VRP by
construction. All t-statistics on means use Newey-West with a 44-day bandwidth;
the test demonstrates the HAC t-stat is a fraction of the OLS one on an
artificially overlapped series. Sample-size honesty: 18 years of daily VRP is
roughly 18 × 12 ≈ 200 *independent* monthly observations, not 4,500.
