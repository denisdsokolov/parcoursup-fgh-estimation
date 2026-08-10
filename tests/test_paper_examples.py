import pandas as pd

from cda_mechanisms import students_proposed


def test_example_2_selects_the_papers_highest_priority_stable_anchor():
    """Reproduce the stable set under anchor a2 = c6 in Example 2."""
    programs = pd.DataFrame(
        {"program_id": [1], "quota": [5], "b_share": [80.0], "r_share": [60.0]}
    ).set_index("program_id", drop=False)
    b_candidates = {3, 5, 6, 8, 10, 12}
    students = pd.DataFrame(
        {
            "student_id": range(1, 13),
            "b_trait": [int(sid in b_candidates) for sid in range(1, 13)],
        }
    ).set_index("student_id", drop=False)
    resident_candidates = [2, 3, 6, 7, 10, 11, 12]
    residents = pd.DataFrame(
        {"student_id": resident_candidates, "program_id": [1] * len(resident_candidates)}
    )

    selected = students_proposed(
        1,
        list(range(1, 13)),
        set(),
        set(range(1, 13)),
        programs,
        students,
        residents,
        "SF-CDA",
    )

    assert set(selected) == {2, 3, 5, 6, 8}
