import pandas as pd
import pytest

from sampling import sample_programmes


def test_sample_programmes_is_reproducible_and_does_not_mutate_input():
    programs = pd.DataFrame(
        {
            "program_id": range(1, 9),
            "kind": [1, 1, 1, 1, 2, 2, 2, 2],
            "b_only_type": [0, 0, 1, 1, 0, 0, 1, 1],
            "br_type": [0] * 8,
        }
    )

    first = sample_programmes(programs, frac=0.5, seed=42)
    second = sample_programmes(programs, frac=0.5, seed=42)

    assert first == second
    assert 0 in first
    assert len(first) == 5
    assert "stratum" not in programs.columns


@pytest.mark.parametrize("frac", [0, -0.1, 1.1])
def test_sample_programmes_validates_fraction(frac):
    programs = pd.DataFrame(
        {"program_id": [1], "kind": [1], "b_only_type": [0], "br_type": [0]}
    )
    with pytest.raises(ValueError, match="frac"):
        sample_programmes(programs, frac=frac)
