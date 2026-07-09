# Methodology notes — realised variance (drafted during Phase 2)

Working notes for the paper's methodology section; every claim here is pinned by a
test in `tests/test_realized_*.py`.

## Why RV estimates quadratic variation

For a semimartingale X, Σᵢ (X_{tᵢ} − X_{tᵢ₋₁})² → [X]_T in probability as the mesh
→ 0. Under GBM, [log S]_T = σ²T; under Heston, [log S]_T = ∫₀ᵀ v_u du — the *path's*
integrated variance, a random variable, which is exactly what a variance swap pays.
So the estimator target is path-specific, and our Heston test checks correlation of
RV with the recorded ∫v dt across paths (>0.999), not just the ensemble mean.

## Why naive RV dies under microstructure noise

Observe Y_i = X_i + u_i, u iid (0, ω²). Observed returns r̃_i = r_i + u_i − u_{i−1},
so E[r̃²] = E[r²] + 2ω² and E[RV_obs] = IV + 2nω². The bias grows linearly in the
number of observations: sampling faster makes it *worse*, which is the volatility
signature plot. At n = 100,000 and ω = 1e-4, the bias term 2nω² = 0.002 dwarfs a
monthly IV of ~0.0007 — the magnitudes in our tests are chosen to make this vivid.

## TSRV: why the two-scale combination works

Slow scale (K subgrids, each with ~n̄ = (n−K+1)/K returns): each subgrid RV carries
noise bias 2n̄ω²; averaging keeps it. Fast scale: bias 2nω². Then
E[avg_slow − (n̄/n)·RV_fast] = (1 − n̄/n)·IV + (2n̄ω² − (n̄/n)·2nω²) = (1 − n̄/n)·IV:
the noise cancels *exactly* in expectation, and dividing by (1 − n̄/n) undoes the
shrinkage of the signal. K ~ n^{2/3} balances the discretisation variance of the slow
scale against the residual noise variance (ZMA 2005).

## Realised kernel: RV plus weighted autocovariances

RK = γ₀ + 2Σ_{h≥1} k(h/(H+1)) γ_h. Noise induces negative first-order autocovariance
in observed returns (an MA(1)); the kernel adds back precisely the autocovariance mass
that the squared terms double-count, with Parzen weights guaranteeing positivity.
H trades noise robustness (larger H) against extra variance; the plug-in rule
H* = c*ξ^{4/5}n^{3/5} with ξ² = ω²/√(T∫σ⁴). Our test accepts recovery within 20%
across a wide band of H — the rule's constant is flagged TODO(verify) in the code.

## Bipower variation: why |r_i||r_{i−1}| is jump-robust

A finite-activity jump contaminates one return (at fine mesh, a.s. isolated). In the
product |r_i||r_{i−1}|, the jump appears multiplied by a *continuous* neighbour of
order √dt → the contribution vanishes in the limit, whereas in r_i² it survives.
Scaling by μ₁⁻² = π/2 corrects E|Z|² vs E[Z²]. Hence BPV → ∫σ²du only, and
RV − BPV → Σ jumps². Our Merton world records each jump, so this decomposition is
tested against the true Σ Y².

## The ratio jump test

z = [(RV−BPV)/RV] / √(θ n⁻¹ max(1, TQ/BPV²)), θ = (π/2)² + π − 5 ≈ 0.609.
The relative jump measure (RV−BPV)/RV is pivotal under H₀ once studentised by the
integrated quarticity ratio; TQ estimates ∫σ⁴ jump-robustly (products of three
adjacent |r|^{4/3}), and TQ/BPV² is scale-free (→1 under constant σ). The max(1,·)
truncation guards the small-sample cases where the quarticity ratio dips below its
theoretical floor. Size and power are verified empirically (800 simulated days each).

## Sampling conventions on real data

Previous-tick sampling on log mid-quotes at 5 minutes (baseline; signature plot
justifies), day = (17:00 ET, 17:00 ET], DST-aware. Held quotes produce zero returns —
they deflate BPV slightly on very quiet days, which is one reason thin days are
flagged rather than silently included. Semivariance splits RV by return sign;
ΔJ = RS⁺ − RS⁻ is the crash-direction conditioning variable for Phase 7.

## Open items carried forward

- TSRV/kernel equation-number citations to be checked against the papers (flagged
  in code as TODO(verify); behaviour pinned by tests either way).
- Weekend gap handling enters at the 30-day *window* level in Phase 4, not in the
  intraday estimators (conventions.md rule 9).
- The panel currently spans whatever the tick backfill has ingested; it extends by
  rerunning `scripts/build_rv_panel.py` and the estimators never change.
