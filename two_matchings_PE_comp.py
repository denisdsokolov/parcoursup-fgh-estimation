import numpy as np
import pandas as pd

def compare_matchings_efficiency(SF_matching: pd.DataFrame,
                                 PCS_matching: pd.DataFrame,
                                 prefs: pd.DataFrame,
                                 students: pd.DataFrame
                                 ) -> tuple[int, int, int, int, int, int, int, int, float, float, float, float]:
    """
    Compare SF-CDA vs CDA (PCS) with three metrics:

    1) Net strict preference:
       (# students who strictly prefer SF) − (# who strictly prefer PCS)

    2) Net additional matches:
       (# matched in SF & unmatched in PCS) − (# matched in PCS & unmatched in SF)

    3) Net average rank improvement (unit steps) among students matched in BOTH:
       mean( rank_PCS − rank_SF | prefer SF ) − mean( rank_SF − rank_PCS | prefer PCS )

    Assumptions:
      - prefs has columns ['student_id','program_id','rank'] with 1 = top choice.
      - SF_matching and PCS_matching have ['student_id','program_id'] (only matched rows).
      - Unmatched students simply do not appear in a matching DataFrame.
      - Unmatched is treated as worse than any listed program (rank = len(list)+1).

    Returns the SF/PCS counts for strict preferences and additional matches,
    followed by the corresponding average rank improvements. Each group of
    metrics is reported for all students and for scholarship holders.
    """

    # --- Build student -> program maps (None if unmatched) ---
    sf_map  = dict(zip(SF_matching['student_id'].astype(int),
                       SF_matching['program_id'].astype(int)))
    pcs_map = dict(zip(PCS_matching['student_id'].astype(int),
                       PCS_matching['program_id'].astype(int)))

    # --- All students we’ll evaluate ---
    all_students = set(prefs['student_id'].astype(int))
    all_students |= set(sf_map.keys())
    all_students |= set(pcs_map.keys())

    stu_b = set(students[students['b_trait'] == 1]['student_id']) # list of all scholarship holders

    # --- Preference positions per student: pos 1=best; outside rank=len(list)+1 ---
    stud_pref_pos: dict[int, dict[int, int]] = {}
    list_len: dict[int, int] = {}
    for sid, sub in prefs.sort_values(['student_id','rank']).groupby('student_id', sort=False):
        progs = sub['program_id'].astype(int).tolist()
        stud_pref_pos[int(sid)] = {p: i for i, p in enumerate(progs, start=1)}
        list_len[int(sid)] = len(progs)

    INF = 10**12

    def rank_pos(sid: int, pid: int | None) -> int:
        """Return rank position for (sid, pid); unmatched or not-in-list = len(list)+1."""
        d = stud_pref_pos.get(sid)
        if d is None:
            return INF  # no list available
        if pid is None:
            return list_len[sid] + 1
        return d.get(pid, list_len[sid] + 1)

    # ---------------- Metric 1: net strict preference ----------------
    prefer_sf = 0
    prefer_pcs = 0
    prefer_sf_b = 0
    prefer_pcs_b = 0
    for s in all_students:
        p_sf  = sf_map.get(s)   # None if unmatched
        p_pcs = pcs_map.get(s)
        if p_sf == p_pcs:
            continue  # identical outcomes → no strict preference

        r_sf  = rank_pos(s, p_sf)
        r_pcs = rank_pos(s, p_pcs)

        if r_sf < r_pcs:
            prefer_sf += 1
            if s in stu_b:
                prefer_sf_b += 1
        elif r_pcs < r_sf:
            prefer_pcs += 1
            if s in stu_b:
                prefer_pcs_b += 1
        # ties (equal ranks) do nothing

    metric1 = prefer_sf - prefer_pcs

    # ---------------- Metric 2: net additional matches ----------------
    sf_only  = sum(1 for s in all_students if s in sf_map  and s not in pcs_map)
    pcs_only = sum(1 for s in all_students if s in pcs_map and s not in sf_map)
    sf_only_b  = sum(1 for s in stu_b if s in sf_map  and s not in pcs_map)
    pcs_only_b = sum(1 for s in stu_b if s in pcs_map and s not in sf_map)
    metric2 = sf_only - pcs_only

    # ---------------- Metric 3: net avg rank improvement --------------
    deltas_sf_pref  = []  # rank_PCS - rank_SF when SF preferred (positive)
    deltas_pcs_pref = []  # rank_SF - rank_PCS when PCS preferred (positive)
    deltas_sf_pref_b  = []  # rank_PCS - rank_SF when SF preferred (positive)
    deltas_pcs_pref_b = []  # rank_SF - rank_PCS when PCS preferred (positive)

    both = set(sf_map.keys()) & set(pcs_map.keys())
    for s in both:
        r_sf  = rank_pos(s, sf_map[s])
        r_pcs = rank_pos(s, pcs_map[s])
        if r_sf == INF or r_pcs == INF:
            continue  # skip if no preference list
        delta = r_pcs - r_sf  # >0 means SF is ranked higher (better)
        if   delta > 0:
            deltas_sf_pref.append(delta)
            if s in stu_b:
                deltas_sf_pref_b.append(delta)
        elif delta < 0:
            deltas_pcs_pref.append(-delta)
            if s in stu_b:
                deltas_pcs_pref_b.append(-delta)

    mean_sf  = float(np.mean(deltas_sf_pref))  if deltas_sf_pref  else 0.0
    mean_pcs = float(np.mean(deltas_pcs_pref)) if deltas_pcs_pref else 0.0
    mean_sf_b  = float(np.mean(deltas_sf_pref_b))  if deltas_sf_pref_b  else 0.0
    mean_pcs_b = float(np.mean(deltas_pcs_pref_b)) if deltas_pcs_pref_b else 0.0
    metric3 = mean_sf - mean_pcs

    return prefer_sf, prefer_pcs, prefer_sf_b, prefer_pcs_b, sf_only, pcs_only, sf_only_b, pcs_only_b, mean_sf, mean_pcs, mean_sf_b, mean_pcs_b
