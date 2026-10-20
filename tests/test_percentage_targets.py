import math
from fractions import Fraction

import pandas as pd
import pytest

from cda_mechanisms import avg_traits_compl, number_of_students_with_je, students_proposed
from percentage_targets import target_count


@pytest.mark.parametrize('share,size,expected', [(25, 80, 20), (25, 100, 25),
    (25, 3, 1), (100, 0, 0), (0, 9, 0), (None, 8, 0),
    (float('nan'), 9, 0), (0.07, 10000, 7), (Fraction(100,3), 3, 1)])
def test_threshold_uses_ceil_on_declared_percentage(share, size, expected):
    assert target_count(share, size) == expected


@pytest.mark.parametrize('share,size', [(-1, 3), (101, 3), (float('inf'), 3), (20, -1), (20, 1.5)])
def test_invalid_policy_parameters_are_rejected(share, size):
    with pytest.raises(ValueError):
        target_count(share, size)


def test_final_enrollment_fulfillment_includes_empty_positive_targets():
    # q=100, n=80, b=20: exactly the 25% target; the empty positive-target
    # program is also fully fulfilled. The r-only program is outside both means.
    programs = pd.DataFrame({'program_id': [1, 2, 3], 'quota': [100, 10, 10],
        'b_share': [25, 50, 0], 'r_share': [50, 50, 100]})
    students = pd.DataFrame({'student_id': range(80), 'b_trait': [1]*20 + [0]*60})
    matching = pd.DataFrame({'student_id': range(80), 'program_id': [1]*80})
    residents = pd.DataFrame({'student_id': range(40), 'program_id': [1]*40})
    assert avg_traits_compl(matching, programs, students, residents) == (1.0, 1.0, 20, 40)


def test_fulfillment_is_unweighted_and_capped_not_a_program_success_indicator():
    programs = pd.DataFrame({'program_id': [1, 2], 'quota': [100, 100],
                             'b_share': [50, 50], 'r_share': [0, 0]})
    matching = pd.DataFrame({'student_id': range(8), 'program_id': [1]*4 + [2]*4})
    students = pd.DataFrame({'student_id': range(8), 'b_trait': [1,0,0,0,1,1,1,1]})
    residents = pd.DataFrame(columns=['student_id', 'program_id'])
    b, r, filled_b, filled_r = avg_traits_compl(matching, programs, students, residents)
    assert b == 0.75  # mean(1/2, min(4/2,1))
    assert math.isnan(r)
    assert (filled_b, filled_r) == (3, 0)


def test_final_envy_uses_actual_cohort_size():
    programs = pd.DataFrame({'program_id': [1], 'quota': [4], 'b_share': [50], 'r_share': [0]})
    students = pd.DataFrame({'student_id': [1,2,3], 'b_trait': [1,0,1]})
    matching = pd.DataFrame({'student_id': [1,2], 'program_id': [1,1]})
    prefs = pd.DataFrame({'student_id': [1,2,3], 'program_id': [1,1,1], 'rank': [1,1,1]})
    ranks = pd.DataFrame({'student_id': [1,2,3], 'program_id': [1,1,1], 'merit_rank': [1,2,3]})
    residents = pd.DataFrame(columns=['student_id', 'program_id'])
    # The b target is already met (1/2); c3 has worse merit than both admits.
    assert number_of_students_with_je(matching, prefs, ranks, programs, students, residents) == (0,0)
    # An unranked candidate cannot claim a seat solely by possessing a trait.
    assert number_of_students_with_je(matching, prefs, ranks.iloc[:2], programs.assign(b_share=100),
                                      students, residents) == (0,0)


def test_residual_targets_above_remaining_capacity_are_saturated():
    programs = pd.DataFrame({'program_id': [1], 'quota': [4], 'b_share': [80], 'r_share': [20]}).set_index('program_id', drop=False)
    students = pd.DataFrame({'student_id': range(6), 'b_trait': [0,0,1,1,1,1]}).set_index('student_id', drop=False)
    residents = pd.DataFrame({'student_id': [0,3,5], 'program_id': [1,1,1]})
    selected = students_proposed(1, list(range(6)), {1,4}, {0,2,3,5}, programs, students, residents, 'SF-CDA')
    assert set(selected) == {2,3}  # {3,5} would leave a merit-justified exchange
