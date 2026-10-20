# Manuscript results

These tables are the public numeric record transcribed from *Stable Choice with Overlapping
Targets: Theory and Evidence from French College Admissions*. They are not computed from the
historical files in `../simulations/`.

- `mechanism_comparison.csv`: main Table 2 (2024), supplemental Tables S4 (2022) and S8 (2023).
  Columns `pcs_sd`, `sf_cda_sd`, and `difference_sd` are across-simulation **standard deviations**;
  differences are paired `SF-CDA - PCS`. Target fulfillment uses final enrollment and is the
  unweighted mean of capped fulfilled-target fractions; see [TARGETS.md](../../docs/TARGETS.md).
- `preference_estimates.csv`: selected coefficients and sample dimensions from supplemental
  Tables S3 (2022), S7 (2023), and S11 (2024). Here `se` denotes coefficient-estimation standard
  errors, a different quantity from simulation SDs. Only selected coefficients are published.

Blank observed entries mean no administrative benchmark is available. Published means are
rounded independently, so subtracting displayed means may differ slightly from a displayed
paired difference. The 2022 paired differences follow the current manuscript: 12,527 filled
b-target positions and 7,722 scholarship holders matched only under the respective mechanism.
The mechanism table carries no simulation-test significance markers, matching the paper.
