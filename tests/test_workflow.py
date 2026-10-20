"""Small end-to-end workflow checks using wholly synthetic processed tables."""
import gzip
from pathlib import Path
import pickle
import runpy

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'FGH_main.py'


def make_inputs(folder):
    folder.mkdir()
    pd.DataFrame({'student_id':[1,2], 'b_trait':[1,0], 'sex':[0,1],
        'is_french':[1,1], 'wishes_count':[2,2], 'b_only_wishes_count':[0,0],
        'br_wishes_resident_count':[1,1]}).to_csv(folder/'data_students.csv',index=False)
    pd.DataFrame({'program_id':[10,20], 'quota':[1,1], 'b_share':[50,50],
        'r_share':[50,50], 'b_only_type':[0,0], 'br_type':[1,1]}).to_csv(folder/'data_programs.csv',index=False)
    pd.DataFrame([(s,f) for s in (1,2) for f in (10,20)],
        columns=['student_id','program_id']).to_csv(folder/'data_wishes.csv',index=False)
    pd.DataFrame({'student_id':[1,2], 'program_id':[10,20]}).to_csv(folder/'data_resident.csv',index=False)
    pd.DataFrame({'student_id':[1,2], 'program_id':[10,20]}).to_csv(folder/'data_matching.csv',index=False)
    pd.DataFrame([(f,s,s,s) for f in (10,20) for s in (1,2)],
        columns=['program_id','student_id','merit_rank','call_rank']).to_csv(folder/'data_ranks.csv',index=False)


@pytest.mark.parametrize('year,cached', [(22,False),(24,False),(22,True),(24,True)])
def test_fresh_and_cached_workflows_write_manuscript_sd_schema(tmp_path,monkeypatch,year,cached):
    folder=tmp_path/'processed';make_inputs(folder)
    monkeypatch.chdir(tmp_path)
    for name,value in {'PARCOURSUP_DATA_DIR':str(folder), 'PARCOURSUP_YEAR':str(year),
        'PARCOURSUP_BUNCH_SIZE':'1', 'PARCOURSUP_BUNCHES':'2',
        'PARCOURSUP_FINAL_ANALYSIS':'true', 'PARCOURSUP_PRINT_STATS':'false',
        'PARCOURSUP_USE_ROLS':str(cached), 'PARCOURSUP_LOAD_ROLS':'true'}.items():
        monkeypatch.setenv(name,value)
    monkeypatch.delenv('PARCOURSUP_RESULT_YEAR',raising=False)
    monkeypatch.delenv('PARCOURSUP_SEED',raising=False)
    if cached:
        with gzip.open(folder/'rols.pkl.gz','wb') as stream:
            pickle.dump({0:[[10,20,0],[20,10,0]],1:[[20,10,0],[10,20,0]]},stream)
        import simulate_rols
        def reject_fresh(*args,**kwargs):
            pytest.fail('cached mode must use the cached profile pool')
        monkeypatch.setattr(simulate_rols,'simulate_ranked_lists',reject_fresh)
    else:
        pd.DataFrame({'program_id':[0,10,20], 'alpha_hat_final':[0,0,0]}).to_csv(folder/'alpha_hat.csv',index=False)
        pd.DataFrame({'name':['near_is'], 'coef':[0]}).to_csv(folder/'beta_hat.csv',index=False)
    runpy.run_path(str(SCRIPT))
    output=folder/f'PaperSummary20{year}.csv'
    first=pd.read_csv(output)
    assert set(['pcs_sd','sf_cda_sd','difference_sd']) <= set(first)
    assert first['simulations'].eq(2).all()
    assert first.loc[first.metric.eq('last_stage_index'),'pcs_mean'].iloc[0]>=1
    runpy.run_path(str(SCRIPT))
    pd.testing.assert_frame_equal(first,pd.read_csv(output))


def test_published_paper_values_follow_current_table_conventions():
    path=SCRIPT.parent/'results/paper/mechanism_comparison.csv'
    data=pd.read_csv(path)
    assert set(['pcs_sd','sf_cda_sd','difference_sd']) <= set(data)
    assert not {'pcs_se','sf_cda_se','difference_se','significance'} & set(data)
    def row(year,metric):return data.loc[(data.year==year)&data.metric.eq(metric)].iloc[0]
    assert row(2024,'matched_scholarship_holders').difference_mean==9796
    assert row(2024,'matched_candidates').difference_mean==272
    assert row(2022,'filled_b_target_positions').difference_mean==12527
    assert row(2022,'scholarship_holders_matched_only').difference_mean==7722


def test_other_year_cannot_silently_reestimate_the_2024_specification(tmp_path,monkeypatch):
    folder=tmp_path/'processed';make_inputs(folder)
    for name,value in {'PARCOURSUP_DATA_DIR':str(folder), 'PARCOURSUP_YEAR':'22',
        'PARCOURSUP_USE_ROLS':'true', 'PARCOURSUP_LOAD_ROLS':'false',
        'PARCOURSUP_PRINT_STATS':'false'}.items():
        monkeypatch.setenv(name,value)
    monkeypatch.delenv('PARCOURSUP_RESULT_YEAR',raising=False)
    with pytest.raises(ValueError,match='2024 specification'):
        runpy.run_path(str(SCRIPT))
