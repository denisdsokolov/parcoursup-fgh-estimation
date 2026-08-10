import gzip
import pickle

import pandas as pd
import pytest

from simulate_rols import simulate_ranked_lists


def _inputs():
    students = pd.DataFrame(
        {
            "student_id": [1, 2],
            "b_trait": [0, 1],
            "wishes_count": [2, 2],
            "sex": [0, 1],
            "is_french": [1, 1],
            "b_only_wishes_count": [0, 1],
            "br_wishes_resident_count": [1, 0],
        }
    )
    programs = pd.DataFrame(
        {
            "program_id": [10, 20],
            "b_only_type": [0, 1],
            "br_type": [1, 0],
            "b_share": [20, 30],
            "r_share": [10, 0],
            "quota": [5, 8],
        }
    )
    wishes = pd.DataFrame(
        {"student_id": [1, 1, 2, 2], "program_id": [10, 20, 10, 20]}
    )
    resident = pd.DataFrame({"student_id": [1], "program_id": [10]})
    alpha_hat = pd.DataFrame(
        {
            "program_id": [0, 10, 20],
            "alpha_hat_final": [0.0, 0.5, -0.25],
        }
    )
    names = [
        "near_is",
        "bOnly_sch",
        "bOnly_bOnlyWish",
        "br_brWishRes",
        "logQuota_nat",
        "logQuota_near",
        "logQuota_sex",
        "logQuota_sch",
        "out_sch_0",
        "out_sex_0",
        "out_nat_0",
        "out_wish_0",
    ]
    beta_hat = pd.DataFrame({"name": names, "coef": [0.0] * len(names)})
    return students, programs, wishes, resident, alpha_hat, beta_hat


def test_simulated_rankings_are_reproducible_and_saved_once(tmp_path):
    inputs = _inputs()
    output = tmp_path / "nested" / "rols.pkl.gz"

    first = simulate_ranked_lists(2, *inputs, seed=123, out_file=output)
    second = simulate_ranked_lists(2, *inputs, seed=123)

    assert first == second
    assert output.exists()
    assert all(ranking[-1] == 0 for draw in first.values() for ranking in draw)
    with gzip.open(output, "rb") as stream:
        assert pickle.load(stream) == first


def test_simulated_rankings_require_a_positive_draw_count():
    with pytest.raises(ValueError, match="R must be at least 1"):
        simulate_ranked_lists(0, *_inputs())
