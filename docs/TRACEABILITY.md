# Paper-to-code traceability and audit

This audit covers every active Python script in the repository. `legacy/` is separately marked
as non-canonical. Status means correspondence with the current manuscript after the fixes in
this repository, not independent validation against the confidential microdata.

| Paper component | Implementation | Status and audit note |
| --- | --- | --- |
| Annual administrative cleaning | `FGH_data_proc.py` | Corresponds to the listed Parcoursup source tables and writes the six documented processed tables. Full verification requires restricted inputs. |
| Load typed processed tables | `load_tables.py` | Consistent with the processed schemas used downstream. |
| Stratified program sample | `sampling.py` | Samples by program category and affirmative-action regime, matching the empirical stratification. Exact replication requires the unpublished saved `progs_keep_set.pkl`. |
| Personalized eligibility set | `panel_builder.py` | Uses the worst admitted `call_rank` as the cutoff for full programs and infinity for undersubscribed programs, then adds the outside option. |
| Logged wish-count interactions | `preference_features.py`, `panel_builder.py` | Uses `log(max(count, 1))`, matching the variables reported in the paper. Earlier repository code divided these two counts by 10; this was corrected. |
| Multinomial-logit estimation | `estimate_fgh.py` | Estimates candidate choice over the personalized feasible set with program constants and interactions. |
| ASC imputation | `alpha_ols.py` | Uses log capacity, target shares, boarding, demand, and program-category indicators as described. |
| Coefficient diagnostics | `model_statistical_significance.py` | Produces parameter uncertainty diagnostics; it is not the source of the simulation-table Monte Carlo errors. |
| Preference simulation | `simulate_rols.py` | Applies every known interaction present in the annual coefficient file plus Gumbel shocks; missing annual coefficients are zero. The outside-option convention follows the empirical run configuration. |
| One profile per market | `pref_profile_sampling.py` | Converts simulated ranked lists into a candidate-program-rank table. |
| Current Parcoursup counterfactual | `cda_mechanisms.py` with `call_rank`, `CDA` | Uses the observed fixed call order. Construction of that administrative order is outside this repository. |
| SF-CDA | `cda_mechanisms.py` with `merit_rank`, `SF-CDA` | Keeps old-timers, recomputes residual capacity and target deficits, counts dual-trait candidates toward both targets, and returns the first stable anchor (`k_left`). Example 2 is a regression test. |
| Justified envy | `number_of_students_with_je` in `cda_mechanisms.py` | Uses merit priority and lexicographically ordered scholarship then residence deficits for one-for-one replacements. |
| Target compliance | `avg_traits_compl` in `cda_mechanisms.py` | Uses ceilings of target share times quota and one-to-all trait counts. |
| Cross-mechanism welfare | `two_matchings_PE_comp.py` | Counts which assignment a candidate prefers, exclusive matches, and rank improvements. |
| Paired experiment | `FGH_main.py` | Passes one preference draw to both mechanisms. Default changed from 500 to the paper's 1,000 markets. |
| Table reporting | `paper_reporting.py` | Reports draw means, Monte Carlo SE (`sd / sqrt(n)`), and paired SF-CDA minus PCS tests. Earlier final reporting used standard deviations; this was corrected. |
| Residency figures | `results/resident_histograms/plot_histograms.py` | Accepts `--year 22`, `23`, or `24`; boarding is not modeled as a separate matching constraint, matching the paper's simulation convention. |

## Explicit replication boundary

The active scripts now implement the mechanism, preference transformations, paired design, and
reporting conventions described in the manuscript. A bit-for-bit reproduction of the paper's
tables cannot be run from the public repository alone because the administrative microdata,
saved sampled program IDs, and estimated coefficients are access-restricted. The CSV files
already under `results/simulations/` are historical development runs (100 or 500 markets in
several cases) and do not numerically equal the manuscript tables. They are retained for
provenance and must not be presented as the canonical paper results.

The authoritative public numeric record is `results/paper/`, transcribed from Tables 5, D.9,
and D.13, with selected annual preference coefficients including entries from D.14. Stage
The paper labels all stage measures as indices. In the implementation, the 95% and 99%
thresholds use zero-based list positions, while `last_iter` is the loop's one-based iteration
counter; the public CSV preserves the manuscript values and labels.

## Automated checks

Synthetic tests cover basic CDA displacement, SF-CDA's selection in paper Example 2, logged
feature construction, paired Monte Carlo reporting, table loading, sampling, profile creation,
configuration, and matching comparisons. They do not replace a confidential-data replication.
