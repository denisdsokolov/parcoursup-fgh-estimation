import pandas as pd
import pytest

from cda_mechanisms import run_cda_fast


def _market():
    programs = pd.DataFrame(
        {
            "program_id": [10, 20],
            "quota": [1, 1],
            "b_share": [0, 0],
            "r_share": [0, 0],
        }
    )
    students = pd.DataFrame({"student_id": [1, 2], "b_trait": [0, 1]})
    residents = pd.DataFrame(columns=["student_id", "program_id"])
    preferences = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 2],
            "program_id": [20, 10, 10, 20],
            "rank": [1, 2, 1, 2],
        }
    )
    ranks = pd.DataFrame(
        {
            "program_id": [10, 10, 20, 20],
            "student_id": [1, 2, 1, 2],
            "call_rank": [1, 2, 1, 2],
            "merit_rank": [1, 2, 1, 2],
        }
    )
    return programs, students, residents, preferences, ranks


def test_cda_respects_student_preferences_after_displacement():
    matching, iterations, filled = run_cda_fast(*_market())

    assert set(matching.itertuples(index=False, name=None)) == {(1, 20), (2, 10)}
    assert iterations == 2
    assert filled == [1, 2]


def test_cda_rejects_unknown_rank_column():
    with pytest.raises(ValueError, match="rank_col"):
        run_cda_fast(*_market(), rank_col="unknown")
