import pandas as pd
import pytest

from cda_mechanisms import run_cda_fast
from paper_reporting import first_filled_stage


@pytest.mark.parametrize('mechanism', ['CDA', 'SF-CDA'])
def test_all_program_choices_precede_candidate_responses(mechanism):
    programs = pd.DataFrame({'program_id': [10,20,30], 'quota': [1]*3,
                             'b_share': [0]*3, 'r_share': [0]*3})
    students = pd.DataFrame({'student_id': [1,2,3], 'b_trait': [0]*3})
    residents = pd.DataFrame(columns=['student_id','program_id'])
    prefs = pd.DataFrame([(1,10,1),(1,20,2),(2,30,1),(2,10,2),(3,20,1)],
                          columns=['student_id','program_id','rank'])
    ranks = pd.DataFrame([(10,2,1),(10,1,2),(20,1,1),(20,3,2),(30,2,1)],
                          columns=['program_id','student_id','merit_rank'])
    ranks['call_rank'] = ranks['merit_rank']
    expected = [{10:frozenset({2}),20:frozenset({1}),30:frozenset({2})},
                {10:frozenset({1}),20:frozenset({1}),30:frozenset({2})},
                {10:frozenset({1}),20:frozenset({3}),30:frozenset({2})}]
    for ordered_programs in (programs, programs.iloc[::-1]):
        history = []
        matching, stages, fills = run_cda_fast(ordered_programs, students, residents, prefs, ranks,
            mechanism=mechanism, proposal_history=history)
        assert history == expected
        assert stages == 3 and fills == [2,2,3]
        assert set(map(tuple, matching.to_numpy())) == {(1,10),(2,30),(3,20)}


def test_stage_numbers_start_at_one():
    assert first_filled_stage([1,9,10], .95) == 3
    assert first_filled_stage([1,9,10], .90) == 2
    assert first_filled_stage([10], .99) == 1
    assert first_filled_stage([0], .99) == 1
    with pytest.raises(ValueError):
        first_filled_stage([], .99)
