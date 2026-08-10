import pandas as pd
import pytest

from load_tables import REQUIRED_TABLES, load_all


def test_load_all_reports_every_missing_table(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        load_all(tmp_path)

    for filename in REQUIRED_TABLES:
        assert filename in str(exc_info.value)


def test_load_all_normalizes_students_and_matching_column(tmp_path):
    pd.DataFrame(
        {"student_id": [1, 1], "post_code": [75001, 75002], "b_trait": [0, 0]}
    ).to_csv(tmp_path / "data_students.csv", index=False)
    pd.DataFrame({"program_id": [10]}).to_csv(
        tmp_path / "data_programs.csv", index=False
    )
    pd.DataFrame({"student_id": [1], "program_id": [10]}).to_csv(
        tmp_path / "data_wishes.csv", index=False
    )
    pd.DataFrame({"student_id": [1], "program_id": [10]}).to_csv(
        tmp_path / "data_resident.csv", index=False
    )
    pd.DataFrame({"student_id": [1], "program_id": [10]}).to_csv(
        tmp_path / "data_matching.csv", index=False
    )
    pd.DataFrame(
        {"student_id": [1], "program_id": [10], "call_rank": [1], "merit_rank": [1]}
    ).to_csv(tmp_path / "data_ranks.csv", index=False)

    students, _, _, _, matches, _ = load_all(tmp_path)

    assert list(students.columns) == ["student_id", "b_trait"]
    assert students.to_dict("records") == [{"student_id": 1, "b_trait": 0}]
    assert "college_id_final" in matches.columns
