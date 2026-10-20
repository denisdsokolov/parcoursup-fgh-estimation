# Historical simulation outputs

These aggregate files are retained development runs for 2022, 2023, and 2024. They are useful
for inspecting output schemas, but they are **not exact replications of the manuscript tables**:
some contain 5, 100, 490, or 500 markets, and multiple same-year files came from different
intermediate specifications.

Use [`../paper/mechanism_comparison.csv`](../paper/mechanism_comparison.csv) for the canonical
values reported in the main and supplemental comparison tables. A newly completed run writes
`PaperSummary{year}.csv` with means and across-simulation standard deviations. The published historical CSVs remain
archival outputs; their stage and target conventions are not interchangeable with a new run.

## Columns

- `PCS_*`, `SF_*`: outcomes under the current fixed call-order mechanism and SF-CDA.
- `*_matched_studs_amount`, `*_matched_b_studs_amount`: all matched candidates and matched
  scholarship holders.
- `*_je_studs_amount`, `*_je_b_studs_amount`: candidates and scholarship holders with at least
  one pairwise justified-envy claim.
- `*_95_iter`, `*_99_iter`: zero-based indices of the first stage crossing each threshold;
  `*_last_iter` is the total number of stages. These historical files mix numbering conventions.
  New runs use one-based stage numbers for all three fields.
- `*_avg_b`, `*_avg_r`: historical program-level target ratios (multiply by 100 for percent). New runs use
  final-enrollment fulfillment; a historical aggregate cannot determine a different denominator
  without the program-level matching.
- `*_total_b`, `*_total_r`: filled scholarship- and residence-target seats.
- `*_stu_pref`, `*_b_stu_pref`: candidates who prefer that mechanism's assignment.
- `*_only_matched`, `*_only_matched_b`: candidates matched only by that mechanism.
- `*_avg_rank_impr`, `*_avg_rank_impr_b`: average preference-rank improvement among relevant
  switchers.
- `NET_*`: `SF_* - PCS_*` when present.

All files are aggregate. Individual-level source rows and identifiers are not published.

The current definitions and populations are in [`docs/TARGETS.md`](../../docs/TARGETS.md).
