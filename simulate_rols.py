import numpy as np
import pandas as pd
import gzip, pickle
from pathlib import Path
from tqdm import tqdm

def simulate_ranked_lists(
    R: int,
    students: pd.DataFrame,
    programs: pd.DataFrame,
    wishes: pd.DataFrame,
    resident: pd.DataFrame,
    alpha_hat: pd.DataFrame,
    beta_hat: pd.DataFrame,
    *,
    out_file: str | Path | None = None,
    use_wishes_only: bool = True, # True: rank only programmes the student wished
    include_outside_rank: bool = False, # False: outside option appended at end; True: outside competes
    seed: int | None = None,
):
    """
    Return a dict {r: [ROL_i for students in the order of 'students']} and (optionally) save it as gz-pickle.

    Utility used for in-market progs (not 0):
        V_is = alpha_s + beta_neat * near_is + beta_near_sch * (near_is * b_i)
    Outside (s = 0), only if include_outside_rank=True:
        U_i0 = alpha_0 + beta_out_sch * b_i + beta_out_wish * wishes_count_i + Gumbel
    """

    if R < 1:
        raise ValueError(f"R must be at least 1; got {R}")

    # 0) Exctracting coeffs
    b = beta_hat.set_index("name")["coef"]
    beta_near = float(b["near_is"])
    #beta_near_sch = float(b["near_sch"])
    beta_bOnly_sch = float(b["bOnly_sch"])
    #beta_br_sch = float(b["br_sch"])
    #beta_br_near = float(b["br_near"])
    #beta_near_nat = float(b["near_nat"])
    #beta_near_sex = float(b["near_sex"])
    #beta_bShare_sex = float(b["bShare_sex"])
    #beta_rShare_sex = float(b["rShare_sex"])
    #beta_rShare_sch = float(b["rShare_sch"])
    #beta_bShare_near = float(b["bShare_near"])
    #beta_bShare_sch = float(b["bShare_sch"])
    #beta_rShare_near = float(b["rShare_near"])
    beta_bOnly_bOnlyWish = float(b["bOnly_bOnlyWish"])
    beta_br_brWishRes = float(b["br_brWishRes"])
    beta_logQuota_nat = float(b["logQuota_nat"])
    beta_logQuota_near = float(b["logQuota_near"])
    beta_logQuota_sex = float(b["logQuota_sex"])
    beta_logQuota_sch = float(b["logQuota_sch"])
    beta_out_sch = float(b["out_sch_0"])
    beta_out_sex = float(b["out_sex_0"])
    beta_out_nat = float(b["out_nat_0"])
    beta_out_wish = float(b["out_wish_0"])

    # 1) Build fast lookups
    # all alphas
    alpha_s = alpha_hat.set_index("program_id")["alpha_hat_final"].astype(float)
    alpha0 = float(alpha_s.loc[0])

    # resident sets per student
    resid_by_sid = resident.groupby("student_id")["program_id"] \
                           .apply(lambda x: set(x.astype(int))) \
                           .to_dict()

    # acceptable progs per student
    if use_wishes_only:
        wishes_by_sid = wishes.groupby("student_id")["program_id"] \
                              .apply(lambda x: np.asarray(x.astype(int))) \
                              .to_dict()
    else:
        # ALL progs except 0 (heavy; use only if needed)
        all_progs_no0 = alpha_s.index[alpha_s.index != 0].to_numpy()

    # Precompute program columns explicitly
    prog = programs.set_index("program_id")
    p_bOnly = prog["b_only_type"].astype(int)
    p_br = prog["br_type"].astype(int)
    p_bShare = (prog["b_share"] / 100.0).astype(float)
    p_rShare = (prog["r_share"] / 100.0).astype(float)
    p_logQuota = np.log(prog["quota"]).astype(float)

    # 2) Iterate students and draw
    rng = np.random.default_rng(seed)
    out = {}

    # Pre-extract the minimal student columns in the order we will output lists
    stu_cols = students[["student_id", "b_trait", "wishes_count", "sex", "is_french", "b_only_wishes_count", "br_wishes_resident_count"]].astype({"student_id": int})

    for r in tqdm(range(R), desc="draw ROLs"):
        lists = []
        for sid, b_i, wcnt, sex, nat, bOnlyWish, brWishRes in stu_cols.itertuples(index=False):
            wcnt = wcnt / 10.0
            bOnlyWish = bOnlyWish / 10.0
            brWishRes = brWishRes / 10.0
            # 2a) the student's considered alternatives
            if use_wishes_only:
                progs = wishes_by_sid.get(sid)
                if progs is None or len(progs) == 0:
                    # no aceptable prog -> only 0
                    lists.append([0])
                    continue
            else:
                progs = all_progs_no0

            # 2b) determenistic utilities for in-market progs
            resid_set = resid_by_sid.get(sid, ())
            # vector of alphas aligned with progs
            a_vec = alpha_s.reindex(progs).to_numpy()

            near = np.fromiter((int(s in resid_set) for s in progs), count=len(progs), dtype=float)
            bOnly = p_bOnly.reindex(progs).to_numpy()
            br = p_br.reindex(progs).to_numpy()
            bShare = p_bShare.reindex(progs).to_numpy()
            rShare = p_rShare.reindex(progs).to_numpy()
            logQuota = p_logQuota.reindex(progs).to_numpy()

            v_is = (a_vec +
                beta_near * near +
                #beta_near_sch * (near * int(b_i)) +
                beta_bOnly_sch * (bOnly * int(b_i)) +
                #beta_br_sch * (br * int(b_i)) +
                #beta_br_near * (br * near) +
                #beta_near_nat * (near * int(nat)) +
                #beta_near_sex * (near * int(sex)) +
                #beta_bShare_sex * (bShare * int(sex)) +
                #beta_rShare_sex * (rShare * int(sex)) +
                #beta_rShare_sch * (rShare * int(b_i)) +
                #beta_bShare_near * (bShare * near) +
                #beta_bShare_sch * (bShare * int(b_i)) +
                #beta_rShare_near * (rShare * near) +
                beta_bOnly_bOnlyWish * (bOnly * bOnlyWish) +
                beta_br_brWishRes * (br * brWishRes) +
                beta_logQuota_nat * (logQuota * int(nat)) +
                beta_logQuota_near * (logQuota * near) +
                beta_logQuota_sex * (logQuota * int(sex)) +
                beta_logQuota_sch * (logQuota * int(b_i)))

            # 2c) add Gumbel shocks and rank descending
            util = v_is + rng.gumbel(size = len(progs))
            order = np.argsort(-util)
            ranked = [int(progs[j]) for j in order]

            # 2d) place outside option
            if include_outside_rank:
                u0 = (alpha0 +
                    beta_out_sch * float(b_i) +
                    beta_out_wish * float(wcnt) +
                    beta_out_sex * float(sex) +
                    beta_out_nat * float(nat) +
                    rng.gumbel())
                # find intersection index to keep descending order
                sorted_utils = util[order]
                # position where u0 would fit (binary search on descending array)
                idx = len(sorted_utils) - np.searchsorted(sorted_utils[::-1], u0, side="left")
                ranked.insert(int(idx), 0)
            else:
                ranked.append(0) # outside at the end by convention

            lists.append(ranked)

        out[r] = lists

    # 3) Save once after all draws are complete. Rewriting the growing object on
    # every iteration is prohibitively expensive for large values of R.
    if out_file is not None:
        output_path = Path(out_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out
