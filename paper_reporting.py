"""Article-style summaries for paired PCS and SF-CDA simulations."""

from __future__ import annotations

from math import sqrt

import pandas as pd
from scipy.stats import ttest_1samp


METRICS = (
    ("avg_b_compliance_pct", "PCS_avg_b", "SF_avg_b", 100.0),
    ("avg_r_compliance_pct", "PCS_avg_r", "SF_avg_r", 100.0),
    ("filled_b_target_seats", "PCS_total_b", "SF_total_b", 1.0),
    ("filled_r_target_seats", "PCS_total_r", "SF_total_r", 1.0),
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


def _standard_error(values: pd.Series) -> float:
    return float(values.std(ddof=1) / sqrt(len(values))) if len(values) > 1 else float("nan")


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def summarize_paired_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired simulations as in Tables 5, D.9, and D.13.

    Standard errors condition on the estimated utility parameters. Difference
    statistics use the paired draw-level difference ``SF-CDA - PCS``.
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
        p_value = float(ttest_1samp(difference, popmean=0).pvalue) if len(difference) > 1 else float("nan")
        rows.append(
            {
                "metric": metric,
                "simulations": len(paired),
                "pcs_mean": float(pcs.mean()),
                "pcs_se": _standard_error(pcs),
                "sf_cda_mean": float(sf.mean()),
                "sf_cda_se": _standard_error(sf),
                "difference_mean": float(difference.mean()),
                "difference_se": _standard_error(difference),
                "p_value": p_value,
                "significance": _stars(p_value),
            }
        )
    return pd.DataFrame(rows)
