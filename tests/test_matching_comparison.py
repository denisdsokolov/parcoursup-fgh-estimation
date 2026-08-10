import pandas as pd

from two_matchings_PE_comp import compare_matchings_efficiency


def test_matching_comparison_counts_preferences_and_rank_gains():
    sf = pd.DataFrame({"student_id": [1, 2], "program_id": [10, 20]})
    pcs = pd.DataFrame({"student_id": [1], "program_id": [20]})
    prefs = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 2],
            "program_id": [10, 20, 20, 10],
            "rank": [1, 2, 1, 2],
        }
    )
    students = pd.DataFrame({"student_id": [1, 2], "b_trait": [1, 0]})

    result = compare_matchings_efficiency(sf, pcs, prefs, students)

    assert result[:8] == (2, 0, 1, 0, 1, 0, 0, 0)
    assert result[8:] == (1.0, 0.0, 1.0, 0.0)
