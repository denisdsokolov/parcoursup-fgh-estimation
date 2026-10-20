# Paper-to-code correspondence

This audit covers the active code against *Stable Choice with Overlapping Targets: Theory and
Evidence from French College Admissions*. The paper's abstract theory uses numerical bounds;
the application and empirical evaluator use percentage targets. `legacy/` contains separately
marked, non-canonical development scripts. Source correspondence and synthetic verification do
not constitute an independent reconstruction of the confidential-data estimates.

| Paper component | Implementation | Correspondence |
| --- | --- | --- |
| Annual administrative cleaning | `FGH_data_proc.py` | Loads annual Parcoursup source tables, preserves policy shares in percent, adjusts simulation seat capacities as described in the paper, and writes six processed tables. Source verification requires the restricted inputs. |
| Typed input tables | `load_tables.py` | Loads the six processed tables and their downstream schemas. |
| Stratified program sample | `sampling.py` | Samples by program category and affirmative-action regime. Exact paper replication requires the saved program IDs. |
| Personalized eligibility set | `panel_builder.py` | Uses the worst admitted call rank for full programs, infinity for underfilled programs, and adds the outside option. |
| Preference features | `preference_features.py`, `panel_builder.py` | Uses logged wish counts, share interactions (shares divided by 100), and log-capacity interactions. |
| MNL estimation | `estimate_fgh.py` | Includes the 2024 baseline specification. Exact 2022/2023 re-estimation needs their original full specifications, which cannot be recovered from selected published coefficients. The workflow guards against silently using the 2024 estimator for another year. |
| Program constants | `alpha_ols.py` | Imputes unsampled constants from capacity, target shares, boarding status, demand, and program category. |
| Coefficient uncertainty | `model_statistical_significance.py` | Reports coefficient standard errors; these are distinct from across-simulation variation. |
| Preference simulation | `simulate_rols.py` | Applies known annual interactions from the supplied coefficient file and Gumbel shocks. The baseline ranks submitted applications above the outside option. |
| One paired profile | `pref_profile_sampling.py`, `FGH_main.py` | Each profile is shared by PCS and SF-CDA. Default runs use 1,000 markets for 2024 and 100 for 2022/2023, unless explicitly overridden. Cached-profile sampling is seeded. |
| PCS | `run_cda_fast` with `call_rank`, `CDA` | Consumes the observed fixed call order. The administrative construction of that order is outside the replication pipeline. |
| Percentage thresholds | `percentage_targets.py` | Computes ceil(alpha times cohort size) from decimal percentage inputs without binary floating-point overshoot. |
| SF-CDA | `students_proposed` | Keeps old-timers, computes the attainable full-cohort size, subtracts held trait counts, caps residual targets at residual capacity, and selects `k_left`. Non-wastefulness selects all newcomers when capacity cannot be filled. |
| Exact integer implementation | `target_basis="stage"` / `"capacity"` | Synthetic tests compare entire proposal histories, not only terminal matches. The equivalence concerns soft targets with seat capacities. |
| Stage protocol | `run_cda_fast` | Fixes all program choices before candidate responses, so all old-timers refer to the same preceding stage. Optional history capture supports verification. |
| Fairness guarantees | `tests/test_percentage_anchor_oracle.py` | Independent exhaustive exchange checks verify residual stability/leftmost anchors, Stage Fairness, non-wastefulness, Top-Trait Fairness, and no-overlap final stability on finite synthetic markets. |
| Final justified envy | `number_of_students_with_je` | Uses merit and lexicographic scholarship/residence deficits at final enrollment. Exchanges preserve cohort size and require program acceptability. |
| Target fulfillment | `avg_traits_compl` | Uses final-enrollment thresholds, capped fulfilled-target fractions, unweighted program means, the manuscript's averaging populations, and the zero-threshold convention. |
| Cross-mechanism welfare | `two_matchings_PE_comp.py` | Counts preferred assignments, exclusive matches, and rank gains, for all candidates and scholarship holders. |
| Table variability | `paper_reporting.py` | Reports means and across-simulation SDs (`*_sd`); the difference SD is calculated from paired SF-CDA-minus-PCS observations. |
| Stage statistics | `first_filled_stage` and returned stage count | New runs number all stages from one. Historical files in `results/simulations/` retain their documented original numbering. |
| Capacity-summary groups | `FGH_main.py` | Uses the rounded criterion T_b(q)+T_r(q)>q; this need not coincide with unrounded shares summing above 100%. |
| Boarding schools | Empirical model and documentation | Treated as regular admission programs. The supplement's separate hard-housing extension is not claimed to be implemented or covered by fixed-capacity equivalence. |
| Residency figures | `results/resident_histograms/plot_histograms.py` | Renders the retained annual aggregate application/residence matrices. These figures do not evaluate quota fulfillment. |

## Published numeric record

`results/paper/` transcribes the current main and supplemental tables. Mechanism-comparison
columns ending in `_sd` are simulation standard deviations; `se` in the preference-coefficient
table denotes estimation standard errors. The two 2022 differences are 12,527 filled
scholarship-target positions and 7,722 exclusive scholarship matches, as in the current paper.

The files in `results/simulations/` remain historical development outputs and are not used to
substitute for the manuscript's reported results. Reproducing the nationwide numbers requires
the restricted administrative microdata, saved sample, and original annual coefficients.

## Verification

Tests use synthetic data and fixed random seeds. They include an independent exchange oracle,
full proposal-history comparisons, the worked anchor example, the two PCS counterexamples,
rounding boundaries, underfilled/empty-cohort fulfillment, synchronous-stage ordering, paired
SDs, data loading, preference simulation, sampling, and workflow smoke tests. No applicant-level
records are needed for these checks.
