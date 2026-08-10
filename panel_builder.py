import pandas as pd
import numpy as np
from preference_features import log_count

def build_panel(students, wishes, resident, matches, keep_set, outside_only_students_share):
    """Return long panelresticted to programmes in keep_set"""
    wishes = wishes[wishes["program_id"].isin(keep_set)].copy()
    resident = resident[resident["program_id"].isin(keep_set)].copy()
    matches = matches[matches["college_id_final"].isin(keep_set)].copy()

    m=matches.copy()
    unsampled = ~m["college_id_final"].isin(keep_set) # identify matches to programmes that were NOT sampled
    m_kept = m.loc[~unsampled] # keep sampled matches as usual
    m_out = m.loc[unsampled, ["student_id"]].assign(college_id_final=0) # for unsampled matches, force outside choice (0) in the restricted model
    truly_unmatched = students.loc[~students["student_id"].isin(m["student_id"]),
                             ["student_id"]].assign(college_id_final=0)
    matches = pd.concat([m_kept, m_out, truly_unmatched], ignore_index = True) # for each unmatched student and student matched to not sampled program, we assign an outside option 0

    resident["near_is"] = 1
    near = (wishes.merge(resident, how="left")
                    .fillna({"near_is":0}).astype({"near_is":"int8"}))

    outside = students[["student_id"]].assign(program_id=0)
    F = pd.concat([wishes,
                    matches.rename(columns={"college_id_final": "program_id"}),
                    outside]).drop_duplicates()

    panel = (F.merge(students, on="student_id")
                .merge(matches, on="student_id", how="left")
                .assign(chosen=lambda d: d["program_id"]==d["college_id_final"])
                .merge(near, on=["student_id", "program_id"], how="left")
                .fillna({"near_is":0}).astype({"near_is":"int8"})
    )

    # We have too many students and want to drop some share of students who have only outside option 0 in the panel
    n_alts = panel.groupby("student_id")["program_id"].nunique() # number of alternatives per each student
    only_out = (
        n_alts.eq(1) & panel.groupby("student_id")["program_id"].max().eq(0)
    ) # is the only alternative the outside option?
    one_choice = panel.groupby("student_id")["chosen"].sum().eq(1) # exactly one chosen row (sanity check)
    outside_only_ids = only_out[only_out & one_choice].index

    # keep a random outside_only_students_share of such students
    keep_some = set(outside_only_ids.to_series().sample(frac = outside_only_students_share, random_state=0))
    mask = ~panel["student_id"].isin(outside_only_ids) | panel["student_id"].isin(keep_some)
    panel = panel[mask]

    assert(panel.groupby("student_id")["chosen"].sum() == 1).all() # checking for MNL req. (one match per student)

    panel["near_sch"] = panel["near_is"] * panel["b_trait"]
    panel["out_sch"] = np.where(panel["program_id"]==0, panel["b_trait"], 0)
    panel["out_wish"] = np.where(panel["program_id"]==0, panel["wishes_count"], 0)
    panel["ASC"] = 1
    return panel.sort_values(["student_id", "program_id"]).reset_index(drop=True)





def build_panel_fgh(students, programs, wishes, resident, matches, ranks, keep_set, outside_only_students_share):
    """Return long panelresticted to programmes in keep_set, using cutoffs"""
    wishes_k = wishes[wishes["program_id"].isin(keep_set)].copy()
    ranks_k = ranks[ranks["program_id"].isin(keep_set)].copy()
    resident_k = resident[resident["program_id"].isin(keep_set)].copy()
    match = matches.rename(columns={"college_id_final": "program_id"})
    match_k = match[match["program_id"].isin(keep_set)].copy()

    # 1) Computing cutoffs for each kept program
    # a) admitted counts per prog
    adm = (match_k.groupby("program_id")
           .size()
           .rename("admitted")
           .reset_index())

    # b) worst admitted call_rank where there ARE admits
    worst = (ranks_k.merge(match_k, on=["student_id", "program_id"], how="inner")
             .groupby("program_id")["call_rank"]
             .max()
             .rename("P_s_worst")
             .reset_index())

    # c) bring in quotas for progs
    meta = programs.loc[programs["program_id"].isin(keep_set), ["program_id", "quota"]].copy()

    # d) combine: meta _ admitted _ worst
    cut = (meta.merge(adm, on="program_id", how="left")
               .merge(worst, on="program_id", how="left"))
    cut["admitted"] = cut["admitted"].fillna(0).astype(int)

    # Set P_s: if admitted < quota -> +inf, else -> P_s_worst
    cut["P_s"] = np.where(cut["admitted"] < cut["quota"], np.inf, cut["P_s_worst"])
    cutoff = cut[["program_id", "P_s"]]

    # 2) Feasible set F
    # merge wishes with call_rank and prog P_s; keep only call_rank <= P_s
    wr = (wishes_k
          .merge(ranks_k[["student_id", "program_id", "call_rank"]],
                 on=["student_id", "program_id"], how="left")
          .merge(cutoff, on="program_id", how="inner"))
    F = (wr.loc[wr["call_rank"] <= wr["P_s"],
                ["student_id", "program_id"]]
            .drop_duplicates())

    m=match_k.copy()
    unsampled = ~m["program_id"].isin(keep_set) # identify matches to programmes that were NOT sampled
    m_kept = m.loc[~unsampled] # keep sampled matches as usual
    m_out = m.loc[unsampled, ["student_id"]].assign(program_id=0) # for unsampled matches, force outside choice (0) in the restricted model
    truly_unmatched = students.loc[~students["student_id"].isin(m["student_id"]),
                             ["student_id"]].assign(program_id=0)
    fin = pd.concat([m_kept, m_out, truly_unmatched], ignore_index = True) # for each unmatched student and student matched to not sampled program, we assign an outside option 0
    fin = fin.rename(columns={"program_id": "final_program_id"})

    outside = students[["student_id"]].assign(program_id=0) # adding one outside row per each student

    F_full = (pd.concat([F,
                    fin.rename(columns={"final_program_id": "program_id"}),
                    outside], ignore_index=True).drop_duplicates())

    resident_k["near_is"] = 1

    panel = (F_full.merge(students, on="student_id")
                .join(programs.set_index("program_id"), on="program_id")
                .merge(fin, on="student_id", how="left")
                .assign(chosen=lambda d: (d["program_id"]==d["final_program_id"]).astype("int8"))
                .merge(resident_k, on=["student_id", "program_id"], how="left")
                .fillna({"near_is":0}).astype({"near_is":"int8"})
    )

    # We have too many students and want to drop some share of students who have only outside option 0 in the panel
    n_alts = panel.groupby("student_id")["program_id"].nunique() # number of alternatives per each student
    only_out = (
        n_alts.eq(1) & panel.groupby("student_id")["program_id"].max().eq(0)
    ) # is the only alternative the outside option?
    one_choice = panel.groupby("student_id")["chosen"].sum().eq(1) # exactly one chosen row (sanity check)
    outside_only_ids = only_out[only_out & one_choice].index

    # keep a random outside_only_students_share of such students
    keep_some = set(outside_only_ids.to_series().sample(frac = outside_only_students_share, random_state=0))
    mask = ~panel["student_id"].isin(outside_only_ids) | panel["student_id"].isin(keep_some)
    panel = panel[mask]

    assert(panel.groupby("student_id")["chosen"].sum() == 1).all() # checking for MNL req. (one match per student)
    # Verify FGH feasibility for non-outside rows (guardrail)
    tmp = (panel[panel["program_id"] != 0]
           .merge(ranks[["student_id", "program_id", "call_rank"]], on = ["student_id", "program_id"], how="left")
           .merge(cutoff, on="program_id", how="left"))
    if not tmp.empty:
        assert (tmp["call_rank"] <= tmp["P_s"]).all(), "Found infeasible (i,s)"

    panel["near_sch"] = panel["near_is"] * panel["b_trait"]

    # new interactions
    panel["bOnly_sch"] = panel["b_only_type"] * panel["b_trait"]
    panel["br_sch"] = panel["br_type"] * panel["b_trait"]
    panel["br_near"] = panel["br_type"] * panel["near_is"]
    panel["near_nat"] = panel["near_is"] * panel["is_french"]
    panel["near_sex"] = panel["near_is"] * panel["sex"]
    panel["bShare_sex"] = (panel["b_share"] / 100.0) * panel["sex"]
    panel["rShare_sex"] = (panel["r_share"] / 100.0) * panel["sex"]
    panel["rShare_sch"] = (panel["r_share"] / 100.0) * panel["b_trait"]
    panel["bShare_near"] = (panel["b_share"] / 100.0) * panel["near_is"]
    panel["bShare_sch"] = (panel["b_share"] / 100.0) * panel["b_trait"]
    panel["rShare_near"] = (panel["r_share"] / 100.0) * panel["near_is"]
    panel["bOnly_bOnlyWish"] = panel["b_only_type"] * log_count(panel["b_only_wishes_count"])
    panel["br_brWishRes"] = panel["br_type"] * log_count(panel["br_wishes_resident_count"])
    panel["logQuota_nat"] = np.log(panel["quota"]) * panel["is_french"]
    panel["logQuota_near"] = np.log(panel["quota"]) * panel["near_is"]
    panel["logQuota_sex"] = np.log(panel["quota"]) * panel["sex"]
    panel["logQuota_sch"] = np.log(panel["quota"]) * panel["b_trait"]

    panel["out_sch"] = np.where(panel["program_id"]==0, panel["b_trait"], 0)
    panel["out_sex"] = np.where(panel["program_id"]==0, panel["sex"], 0)
    panel["out_nat"] = np.where(panel["program_id"]==0, panel["is_french"], 0)
    panel["out_wish"] = np.where(panel["program_id"]==0, (panel["wishes_count"] / 10.0), 0)
    panel["ASC"] = 1
    return panel.sort_values(["student_id", "program_id"]).reset_index(drop=True)
