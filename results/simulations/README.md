# Simulation results

These files contain aggregate results for two matching mechanisms on simulated markets for
2022, 2023, and 2024 (the Parcoursup main admission stage with simulated student preferences).

## Columns

The tables contain the following columns:
PCS_matched_studs_amount - number of matched students under PCS,
SF_matched_studs_amount - same for SF-CDA,
PCS_matched_b_studs_amount - number of matched boursiers under PCS,
SF_matched_b_studs_amount - same for SF-CDA,
PCS_je_studs_amount - number of blocking students under PCS,
SF_je_studs_amount - same for SF-CDA,
PCS_je_b_studs_amount - number of blocking boursiers under PCS,
SF_je_b_studs_amount - same for SF-CDA,
PCS_95_iter - index of iteration when at least 95% of seats are filled,
PCS_99_iter - index of iteration when at least 99% of seats are filled,
PCS_last_iter - index of the last iteration,
SF_95_iter - same for SF-CDA,
SF_99_iter - same for SF-CDA,
SF_last_iter - same for SF-CDA,
PCS_avg_b - average compliance of b target seats,
SF_avg_b - same for SF-CDA,
PCS_avg_r - average compliance of r target seats,
SF_avg_r - same for SF-CDA,
PCS_total_b - total fill for b target seats,
SF_total_b - same for SF-CDA,
PCS_total_r - total fill for r target seats,
SF_total_r - same for SF-CDA,
PCS_stu_pref - number of students that prefer PCS matching,
SF_stu_pref - same for SF-CDA,
PCS_b_stu_pref - number of boursiers that prefer PCS matching,
SF_b_stu_pref - same for SF-CDA,
PCS_only_matched - number of students that matched only in PCS matching,
SF_only_matched - same for SF-CDA,
PCS_only_matched_b - number of boursiers that matched only in PCS matching,
SF_only_matched_b - same for SF-CDA,
PCS_avg_rank_impr - average rank improvement if switched to PCS matching,
SF_avg_rank_impr - same for SF-CDA
PCS_avg_rank_impr_b - average rank improvement for boursiers if switched to PCS matching,
SF_avg_rank_impr_b - same for SF-CDA

`NET_*` columns are calculated as `SF_* - PCS_*`.

The calculations use approximately 880,000 candidates and 21,000 programs per year. Source
tables: `a_adm`, `a_rec_grp`, `a_rec`, `a_voe`, `c_can_grp`, `g_can`, `g_tri_ins`, `i_ins`,
and `sp_g_tri_aff`.
