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


def test_pcs_one_program_percentage_counterexample():
    from fractions import Fraction
    from cda_mechanisms import run_cda_fast, number_of_students_with_je
    programs = pd.DataFrame({'program_id':[1], 'quota':[3],
        'b_share':[Fraction(100,3)], 'r_share':[Fraction(100,3)]})
    students = pd.DataFrame({'student_id':[1,2,3,4], 'b_trait':[1,0,0,1]})
    residents = pd.DataFrame({'student_id':[2,4], 'program_id':[1,1]})
    prefs = pd.DataFrame({'student_id':[1,2,3,4], 'program_id':[1]*4, 'rank':[1]*4})
    ranks = pd.DataFrame({'student_id':[1,2,3,4], 'program_id':[1]*4,
                          'merit_rank':[1,2,3,4], 'call_rank':[2,3,4,1]})
    matching,_,_=run_cda_fast(programs,students,residents,prefs,ranks)
    assert set(matching['student_id']) == {1,2,4}
    assert number_of_students_with_je(matching,prefs,ranks,programs,students,residents)==(1,0)


def test_pcs_two_program_top_trait_counterexample_and_sf_repair():
    from cda_mechanisms import run_cda_fast, number_of_students_with_je
    programs = pd.DataFrame({'program_id':[1,2], 'quota':[2,3], 'b_share':[50,0], 'r_share':[0,0]})
    students = pd.DataFrame({'student_id':[1,2,3,4,5], 'b_trait':[1,0,1,0,1]})
    residents = pd.DataFrame(columns=['student_id','program_id'])
    prefs = pd.DataFrame([(s,f,k) for s in range(1,6) for k,f in
        enumerate([2,1] if s in (1,3) else [1,2],1)],columns=['student_id','program_id','rank'])
    ranks = pd.DataFrame([(f,s,s,s) for f in (1,2) for s in range(1,6)],
                          columns=['program_id','student_id','merit_rank','call_rank'])
    pcs,_,_=run_cda_fast(programs,students,residents,prefs,ranks)
    sf,_,_=run_cda_fast(programs,students,residents,prefs,ranks,'merit_rank','SF-CDA')
    assert set(pcs.loc[pcs.program_id==1,'student_id'])=={2,4}
    assert number_of_students_with_je(pcs,prefs,ranks,programs,students,residents)==(1,1)
    assert number_of_students_with_je(sf,prefs,ranks,programs,students,residents)==(0,0)
