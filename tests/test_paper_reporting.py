import pandas as pd
import pytest

from paper_reporting import METRICS, summarize_paired_results


def _complete_result_frame():
    data = {}
    for _, pcs_column, sf_column, _ in METRICS:
        data[pcs_column] = [1.0, 2.0, 3.0]
        data[sf_column] = [2.0, 3.1, 3.9]
    return pd.DataFrame(data)


def test_paper_summary_uses_paired_differences_and_monte_carlo_se():
    summary = summarize_paired_results(_complete_result_frame())
    matched = summary.loc[summary["metric"] == "matched_candidates"].iloc[0]

    assert matched["simulations"] == 3
    assert matched["pcs_mean"] == 2.0
    assert matched["pcs_se"] == pytest.approx(1 / 3**0.5)
    assert matched["difference_mean"] == 1.0
    assert matched["difference_se"] == pytest.approx(0.1 / 3**0.5)
    assert matched["significance"] == "**"


def test_paper_summary_scales_compliance_to_percentage_points():
    summary = summarize_paired_results(_complete_result_frame())
    compliance = summary.loc[summary["metric"] == "avg_b_compliance_pct"].iloc[0]

    assert compliance["pcs_mean"] == 200.0
    assert compliance["difference_mean"] == 100.0


def test_paper_summary_rejects_incomplete_results():
    with pytest.raises(ValueError, match="Missing columns"):
        summarize_paired_results(pd.DataFrame())
