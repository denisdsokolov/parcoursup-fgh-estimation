"""Article-style summaries for paired PCS and SF-CDA simulations."""

from __future__ import annotations

import pandas as pd


METRICS = (
    ("avg_b_target_fulfillment_pct", "PCS_avg_b", "SF_avg_b", 100.0),
    ("avg_r_target_fulfillment_pct", "PCS_avg_r", "SF_avg_r", 100.0),
    ("filled_b_target_positions", "PCS_total_b", "SF_total_b", 1.0),
    ("filled_r_target_positions", "PCS_total_r", "SF_total_r", 1.0),
    ("candidates_with_justified_envy", "PCS_je_studs_amount", "SF_je_studs_amount", 1.0),
    ("scholarship_holders_with_justified_envy", "PCS_je_b_studs_amount", "SF_je_b_studs_amount", 1.0),
    ("matched_candidates", "PCS_matched_studs_amount", "SF_matched_studs_amount", 1.0),
    ("matched_scholarship_holders", "PCS_matched_b_studs_amount", "SF_matched_b_studs_amount", 1.0),
    ("last_stage_index", "PCS_last_iter", "SF_last_iter", 1.0),
    ("stage_99_filled_index", "PCS_99_iter", "SF_99_iter", 1.0),
    ("stage_95_filled_index", "PCS_95_iter", "SF_95_iter", 1.0),
    ("candidates_preferring_mechanism", "PCS_stu_pref", "SF_stu_pref", 1.0),
    ("scholarship_holders_preferring_mechanism", "PCS_b_stu_pref", "SF_b_stu_pref", 1.0),
    ("candidates_matched_only", "PCS_only_matched", "SF_only_matched", 1.0),
    ("scholarship_holders_matched_only", "PCS_only_matched_b", "SF_only_matched_b", 1.0),
    ("avg_rank_improvement", "PCS_avg_rank_impr", "SF_avg_rank_impr", 1.0),
    ("scholarship_holders_avg_rank_improvement", "PCS_avg_rank_impr_b", "SF_avg_rank_impr_b", 1.0),
)


def first_filled_stage(filled_per_stage: list[int], fraction: float) -> int:
    """First one-based stage attaining a fraction of eventual matches."""
    if not filled_per_stage or not 0 < fraction <= 1:
        raise ValueError("a nonempty stage history and a fraction in (0, 1] are required")
    target = fraction * filled_per_stage[-1]
    return next(k for k, count in enumerate(filled_per_stage, start=1) if count >= target)


def summarize_paired_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the main and supplemental mechanism-comparison tables.

    Parentheses in the paper are across-simulation standard deviations,
    conditional on the estimated utility parameters. Difference statistics
    use the paired draw-level difference ``SF-CDA - PCS``. These are not
    standard errors of the estimated means or preference-parameter errors.
    """
    rows = []
    for metric, pcs_column, sf_column, scale in METRICS:
        missing = {pcs_column, sf_column} - set(results.columns)
        if missing:
            raise ValueError(f"Missing columns for {metric}: {sorted(missing)}")

        paired = results[[pcs_column, sf_column]].dropna().astype(float) * scale
        pcs = paired[pcs_column]
        sf = paired[sf_column]
        difference = sf - pcs
        rows.append(
            {
                "metric": metric,
                "simulations": len(paired),
                "pcs_mean": float(pcs.mean()),
                "pcs_sd": float(pcs.std(ddof=1)),
                "sf_cda_mean": float(sf.mean()),
                "sf_cda_sd": float(sf.std(ddof=1)),
                "difference_mean": float(difference.mean()),
                "difference_sd": float(difference.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)
