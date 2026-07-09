# FXE early-exercise premium (American - European) / European

CRR binomial, continuous EUR-yield approximation of FXE distributions; 500 steps. Grid: K/S in (0.97, 0.99, 1.0, 1.01, 1.03), vol in (0.06, 0.09, 0.12).

| regime | tenor | ATM put | ATM call | worst corner |
|---|---|---|---|---|
| 2015-21 (r=0.25%, q=-0.50%) | 30d | 0.35% | 0.00% | 1.19% |
| 2015-21 (r=0.25%, q=-0.50%) | 60d | 0.53% | 0.00% | 1.50% |
| current (r=3.60%, q=2.20%) | 30d | 0.82% | 0.00% | 3.02% |
| current (r=3.60%, q=2.20%) | 60d | 1.33% | 0.00% | 3.95% |
| 2023 peak (r=5.50%, q=3.90%) | 30d | 1.01% | 0.00% | 3.73% |
| 2023 peak (r=5.50%, q=3.90%) | 60d | 1.65% | 0.00% | 4.97% |
| inverted (r=2.00%, q=4.00%) | 30d | 0.00% | 1.28% | 4.75% |
| inverted (r=2.00%, q=4.00%) | 60d | 0.00% | 2.07% | 6.57% |

Worst case across the grid: **6.57%** of option value.

Decision (standing): premiums of this size cannot be waved away, so FXE IV extraction de-Americanizes against the binomial; OTM-only MFIV keeps the residual bias near or below ~1%.
