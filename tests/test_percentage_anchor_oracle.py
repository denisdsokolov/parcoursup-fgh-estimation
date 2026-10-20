"""Independent finite checks of the paper's exchange definition."""
from itertools import combinations
from random import Random
from math import ceil

import pandas as pd
import pytest

from cda_mechanisms import number_of_students_with_je, run_cda_fast, students_proposed


def stable_choices(order, old, capacity, traits, shares):
    """Enumerate full residual cohorts and reject every blocking exchange."""
    pool = set(order) - old
    size = min(capacity-len(old), len(pool))
    thresholds = [ceil(share * (len(old)+size) / 100) for share in shares]
    merit = {s:i for i,s in enumerate(order)}
    def deficits(cohort):
        return tuple(max(T-sum(traits[s][t] for s in cohort),0) for t,T in enumerate(thresholds))
    choices = []
    for seq in combinations(sorted(pool), size):
        selected = set(seq)
        before = deficits(old | selected)
        blocked = False
        for incoming in pool-selected:
            for outgoing in selected:
                after = deficits(old | (selected-{outgoing}) | {incoming})
                if after < before or (after == before and merit[incoming] < merit[outgoing]):
                    blocked = True
        if not blocked:
            choices.append(selected)
    return choices


def test_residual_selector_matches_exhaustive_exchange_oracle():
    rng = Random(20261020)
    for _ in range(200):
        n = rng.randint(2,7); capacity = rng.randint(1,n+2)
        old = set(rng.sample(range(n), rng.randint(0,min(n,capacity-1))))
        traits = {s:(rng.randrange(2),rng.randrange(2)) for s in range(n)}
        shares = rng.choice([0,20,50,80,100]), rng.choice([0,20,50,80,100])
        programs = pd.DataFrame({'program_id':[1], 'quota':[capacity],
            'b_share':[shares[0]], 'r_share':[shares[1]]}).set_index('program_id',drop=False)
        students = pd.DataFrame({'student_id':range(n), 'b_trait':[traits[s][0] for s in range(n)]}).set_index('student_id',drop=False)
        residents = pd.DataFrame([(s,1) for s in range(n) if traits[s][1]],columns=['student_id','program_id'])
        selected = set(students_proposed(1,list(range(n)),old,set(range(n))-old,
                                         programs,students,residents,'SF-CDA'))
        oracle = stable_choices(list(range(n)),old,capacity,traits,shares)
        assert selected in oracle
        assert sum(all(traits[s]) for s in selected) == min(sum(all(traits[s]) for s in S) for S in oracle)


@pytest.mark.parametrize('no_overlap', [False, True])
def test_entire_proposal_histories_and_final_fairness(no_overlap):
    rng = Random(20261020)
    for _ in range(35):
        n = rng.randint(3,7); nf = rng.randint(1,3)
        bs = [rng.randrange(2) for _ in range(n)]
        traits = [{s:(bs[s],0 if no_overlap and bs[s] else rng.randrange(2)) for s in range(n)} for f in range(nf)]
        shares = [(rng.choice([0,20,50,80,100]),rng.choice([0,20,50,80,100])) for f in range(nf)]
        quotas = [rng.randint(1,n+1) for f in range(nf)]
        orders = [rng.sample(range(n),n) for f in range(nf)]
        programs = pd.DataFrame({'program_id':range(nf),'quota':quotas,
            'b_share':[x[0] for x in shares],'r_share':[x[1] for x in shares]})
        students = pd.DataFrame({'student_id':range(n),'b_trait':bs})
        residents = pd.DataFrame([(s,f) for f in range(nf) for s in range(n) if traits[f][s][1]],columns=['student_id','program_id'])
        prefs = pd.DataFrame([(s,f,k) for s in range(n) for k,f in enumerate(rng.sample(range(nf),nf),1)],columns=['student_id','program_id','rank'])
        ranks = pd.DataFrame([(f,s,k) for f in range(nf) for k,s in enumerate(orders[f],1)],columns=['program_id','student_id','merit_rank'])
        history=[]; integer_history=[]
        matching,stages,fills=run_cda_fast(programs,students,residents,prefs,ranks,'merit_rank','SF-CDA',proposal_history=history)
        other,_,_=run_cda_fast(programs,students,residents,prefs,ranks,'merit_rank','SF-CDA',target_basis='capacity',proposal_history=integer_history)
        assert history == integer_history
        pd.testing.assert_frame_equal(matching,other)
        # Compare every stage to an independently enumerated stable residual problem.
        old={f:set() for f in range(nf)}; available={f:set(range(n)) for f in range(nf)}
        pref_rank={(r.student_id,r.program_id):r.rank for r in prefs.itertuples()}
        for choices in history:
            for f in range(nf):
                pool=[s for s in orders[f] if s in available[f]]
                oracle=stable_choices(pool,old[f],quotas[f],traits[f],shares[f])
                selected=set(choices[f])-old[f]
                assert old[f] <= choices[f] and selected in oracle
                assert sum(all(traits[f][s]) for s in selected) == min(sum(all(traits[f][s]) for s in S) for S in oracle)
            next_old={f:set() for f in range(nf)}
            for s in range(n):
                offers=[f for f in range(nf) if s in choices[f]]
                if offers:
                    kept=min(offers,key=lambda f:pref_rank[s,f]);next_old[kept].add(s)
                    for f in offers:
                        if f!=kept:available[f].remove(s)
            old=next_old
        for f in range(nf):
            cohort=old[f]
            for s in available[f]-cohort:
                assigned=next((g for g in range(nf) if s in old[g]),None)
                if assigned is None or pref_rank[s,f] < pref_rank[s,assigned]:
                    assert len(cohort)==quotas[f]  # non-wastefulness
                    threshold=ceil(shares[f][0]*len(cohort)/100)
                    if bs[s]:assert sum(bs[c] for c in cohort)>=threshold  # Top-Trait Fairness
        if no_overlap:
            assert number_of_students_with_je(matching,prefs,ranks,programs,students,residents)==(0,0)
