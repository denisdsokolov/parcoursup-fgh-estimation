# Percentage targets and implementation

The application in *Stable Choice with Overlapping Targets: Theory and Evidence from French
College Admissions* takes minimum target shares as primitives. Its abstract local-choice
results continue to use numerical lower and upper targets. The empirical implementation here
uses seat capacities and two ordered soft lower targets, with scholarship (`b`) before
residence (`r`). Input columns `b_share` and `r_share` are **percentages in [0, 100]**;
manuscript alpha is that value divided by 100. A dual-trait candidate counts toward both.

## Cohort thresholds

For a cohort D, `T_t(n) = ceil(alpha_t * n)` and
`Delta_t(D) = max(T_t(len(D)) - number_with_trait_t(D), 0)`.
`percentage_targets.target_count` performs the ceiling using the declared decimal percentage,
avoiding floating-point overshoot at exact integers. Missing shares mean zero targets.

`students_proposed` keeps the old-timer set O fixed and chooses from the remaining available
newcomers N. With `m = quota - len(O)`, its attainable full cohort has size
`s = len(O) + min(m, len(N))`. Its raw residual target is
`r_t = max(T_t(s) - number_with_trait_t(O), 0)`; the numerical anchor instance uses
`min(m, r_t)`. Saturating at m subtracts the same constant from every deficit, preserving
all one-for-one comparisons. The selected anchor is the first stable anchor (`k_left`).

The paper's **Exact integer implementation** proposition shows that replacing T_t(s) by
T_t(quota) gives exactly the same choices, proposals, rejections, and final matching. If enough
newcomers remain, s equals capacity. Otherwise non-wastefulness selects every newcomer.
`target_basis="stage"` is the default; `target_basis="capacity"` exposes the equivalent
integer implementation for verification. Both saturate residual targets at m.

## Stages and fairness

`run_cda_fast` freezes every program's choices before processing candidate responses. Thus all
old-timers come from the same preceding tentative matching, independently of program iteration
order. Candidates retain their best offer, and the process ends at a stage with no rejection.
Stage numbers start at **one**, including the 95%, 99%, and last-stage statistics.

Synthetic checks independently enumerate stable residual cohorts, verify the left-endpoint
selection, and compare complete proposal histories between the two implementations. They also
check Stage Fairness, non-wastefulness, Top-Trait Fairness, full stability without overlap,
and the two PCS counterexamples from the paper. Final justified envy uses final enrollment
and the unchanged-size one-for-one exchange definition.

## Final-enrollment fulfillment

For each final cohort with n admits and n_t candidates of trait t,

- filled target positions: `H_t = min(n_t, T_t(n))`;
- fulfillment: `F_t = H_t / T_t(n)` if T_t(n)>0, otherwise **1**.

`avg_traits_compl` retains its historical function name for callers and returns
`(mean_b, mean_r, sum_H_b, sum_H_r)`. Means are **unweighted**, capped fractions in [0, 1].
The b population comprises programs with positive b shares; the r population comprises
programs with both positive shares. Empty cohorts with positive policy shares remain in the
means, with fulfillment 1. A mean of fractions is distinct from the share of programs fully
meeting their targets. The raw output fields `PCS_avg_b`, `SF_avg_b`, etc. retain their names;
`paper_reporting` labels their meaning as target fulfillment.

Capacity-induced counts T_t(quota) still describe the program-summary tables and identify the
group `T_b(quota) + T_r(quota) > quota`. Because of rounding, that group is not necessarily
identical to programs whose unrounded shares sum to more than 100%.

## Simulation variability and scope

Mechanism-comparison tables report **across-simulation standard deviations** (`*_sd`), including
the SD of paired SF-CDA-minus-PCS differences. They condition on estimated utility parameters.
These SDs are distinct from Monte Carlo standard errors of means and from the coefficient
standard errors (`se`) in preference-estimation tables.

The empirical counterfactual treats boarding programs as regular admissions programs. Housing
status may enter preference estimation, but `run_cda_fast` has no hard dormitory-capacity
constraint. The supplement's housing extension uses its own maximum feasible cohort size and
is not implemented here; fixed-capacity equivalence is not asserted for that extension.
