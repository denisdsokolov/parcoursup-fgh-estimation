import pandas as pd
import pytest

from pref_profile_sampling import sample_random_profile


def test_profile_sampling_validates_draw_shape():
    students = pd.DataFrame({"student_id": [1, 2]})

    with pytest.raises(ValueError, match="at least one"):
        sample_random_profile({}, students)

    with pytest.raises(ValueError, match="one list per student"):
        sample_random_profile({0: [[10, 0]]}, students)


def test_profile_sampling_drops_outside_option_and_preserves_order():
    students = pd.DataFrame({"student_id": [1, 2]})
    rols = {0: [[20, 10, 0], [30, 0]]}

    profile = sample_random_profile(rols, students, seed=5)

    assert profile.to_dict("records") == [
        {"student_id": 1, "program_id": 20, "rank": 1},
        {"student_id": 1, "program_id": 10, "rank": 2},
        {"student_id": 2, "program_id": 30, "rank": 1},
    ]
