from pathlib import Path
import gzip, pickle, gc
import load_tables, sampling, panel_builder, estimate_fgh, alpha_ols, simulate_rols, pref_profile_sampling, cda_mechanisms, model_statistical_significance, two_matchings_PE_comp
import pandas as pd
import numpy as np
from project_config import env_bool, env_int, env_path

gc.collect() # frees up memory

# Choosing setting: Yc - we exclude candidates that refused all propositions and remained unmatched suring main adm. phase
#                   Yq - we exclude all empty seats for programs that were wished my an unmatched candidate
excl_cand = env_bool("PARCOURSUP_EXCLUDE_CANDIDATES", True)
excl_seats = env_bool("PARCOURSUP_EXCLUDE_SEATS", True)
if excl_cand:
    tag_c = "Yc"
else:
    tag_c = "Nc"

if excl_seats:
    tag_q = "Yq"
else:
    tag_q = "Nq"

DATADIR = env_path("PARCOURSUP_DATA_DIR", "FGH_data" + "_" + tag_c + "_" + tag_q) # folder to read data end save estimations
p = Path(DATADIR)
folder = p

orig_stat_print = env_bool("PARCOURSUP_PRINT_STATS", False) # Do we want to print initial data stats?

use_ROLs = env_bool("PARCOURSUP_USE_ROLS", False) # Do we want to use heavy ROLs file or create new ROLs per each market?
load_ROLs = env_bool("PARCOURSUP_LOAD_ROLS", True) # True is ROLs of students are already constructed and saved

bunch_size = env_int("PARCOURSUP_BUNCH_SIZE", 10, minimum=1) # how many markets in one bunch
bunches_number = env_int("PARCOURSUP_BUNCHES", 50, minimum=1) # number of bunches
markets_number = bunch_size*bunches_number # how many simulated markets
simulation_seed = env_int("PARCOURSUP_SEED", 2024, minimum=0)
result_year = env_int("PARCOURSUP_RESULT_YEAR", 2024, minimum=2000)

final_analysis = env_bool("PARCOURSUP_FINAL_ANALYSIS", True) # after all simulated data is obtained

# Loading data
students, programs, wishes, resident, matches, ranks = load_tables.load_all(DATADIR)

# Original statistics
if orig_stat_print:
    # Numbers of diff progs type:
    print("================= " + tag_c + "_" + tag_q + " Descriptive stats =========================")
    progs_num = programs["program_id"].nunique()
    progs_q = programs["quota"].sum()
    print(f"Programs #: {progs_num}; total quota = {progs_q}")

    internat_num = len(programs[programs["internat"] == 1])
    internat_q = programs.loc[programs["internat"] == 1, "quota"].sum()
    print(f" - Internats #: {internat_num} ({100*internat_num/progs_num:.1f}%); total quota = {internat_q} ({100*internat_q/progs_q:.1f}%)")

    b_only_progs_num = len(programs[programs["b_only_type"] == 1])
    b_only_progs_q = programs.loc[programs["b_only_type"] == 1, "quota"].sum()
    print(f" - b-only #: {b_only_progs_num} ({100*b_only_progs_num/progs_num:.1f}%); total quota = {b_only_progs_q} ({100*b_only_progs_q/progs_q:.1f}%)")
    if len(programs[(programs["b_only_type"] == 0) & (programs["internat"] == 1)]) == 0:
        print("       All internats are b-only programs!")

    br_progs_num = len(programs[programs["br_type"] == 1])
    br_progs_q = programs.loc[programs["br_type"] == 1, "quota"].sum()
    print(f" - br #: {br_progs_num} ({100*br_progs_num/progs_num:.1f}%); total quota = {br_progs_q} ({100*br_progs_q/progs_q:.1f}%)")

    r_only_progs_num = len(programs[(programs["b_share"] == 0) & (programs["r_share"] > 0)])
    r_only_progs_q = programs.loc[(programs["b_share"] == 0) & (programs["r_share"] > 0), "quota"].sum()
    print(f" - r-only #: {r_only_progs_num} ({100*r_only_progs_num/progs_num:.1f}%); total quota = {r_only_progs_q} ({100*r_only_progs_q/progs_q:.1f}%)")

    no_AA_progs_num = progs_num - b_only_progs_num - br_progs_num - r_only_progs_num
    no_AA_progs_q = progs_q - b_only_progs_q - br_progs_q - r_only_progs_q
    print(f" - no-AA #: {no_AA_progs_num} ({100*no_AA_progs_num/progs_num:.1f}%); total quota = {no_AA_progs_q} ({100*no_AA_progs_q/progs_q:.1f}%)")

    # Students stats
    print("------------------------")
    stu_num = students['student_id'].nunique()
    print(f"Candidates #: {stu_num}")

    stu_b_num = len(students[students["b_trait"] == 1])
    print(f" - with trait b: {stu_b_num} ({100*stu_b_num/stu_num:.1f}%)")

    stu_fem_num = len(students[students["sex"] == 1])
    print(f" - females: {stu_fem_num} ({100*stu_fem_num/stu_num:.1f}%)")
    stu_fem_b_num = len(students[(students["sex"] == 1) & (students["b_trait"] == 1)])
    print(f"   - females with b: {stu_fem_b_num} ({100*stu_fem_b_num/stu_fem_num:.1f}% of females)")

    stu_mal_num = len(students[students["sex"] == 0])
    print(f" - males: {stu_mal_num} ({100*stu_mal_num/stu_num:.1f}%)")
    stu_mal_b_num = len(students[(students["sex"] == 0) & (students["b_trait"] == 1)])
    print(f"   - males with b: {stu_mal_b_num} ({100*stu_mal_b_num/stu_mal_num:.1f}% of males)")

    stu_fr_num = len(students[students["is_french"] == 1])
    print(f" - french: {stu_fr_num} ({100*stu_fr_num/stu_num:.1f}%)")
    stu_fr_b_num = len(students[(students["is_french"] == 1) & (students["b_trait"] == 1)])
    print(f"   - french with b: {stu_fr_b_num} ({100*stu_fr_b_num/stu_fr_num:.1f}% of french)")

    stu_notfr_num = len(students[students["is_french"] == 0])
    print(f" - not french: {stu_notfr_num} ({100*stu_notfr_num/stu_num:.1f}%)")
    stu_notfr_b_num = len(students[(students["is_french"] == 0) & (students["b_trait"] == 1)])
    print(f"   - not french with b: {stu_notfr_b_num} ({100*stu_notfr_b_num/stu_notfr_num:.1f}% of not french)")

    # Original matching statistics:
    print("------------------------")
    original_matching = matches.rename(columns={"college_id_final": "program_id"})
    stu_matched = len(original_matching)
    print(f"Candidates init. matched: {stu_matched} ({(100*stu_matched/stu_num):.1f}% of candidates; {(100*stu_matched/progs_q):.1f}% of quota)")
    stu_b = students[students['b_trait'] == 1]['student_id']
    orig_match_b = matches[matches['student_id'].isin(stu_b)]
    stu_b_matched = len(orig_match_b)
    print(f"Sch. holders init. matched: {stu_b_matched} ({(100*stu_b_matched/stu_b_num):.1f}% of sch. holders)")
    original_avg_b, original_avg_r, original_total_b, original_total_r = cda_mechanisms.avg_traits_compl(original_matching, programs, students, resident)
    print(f"Initial matching avg b- and r-compliances: ({100*original_avg_b:.2f}%, {100*original_avg_r:.2f}%)")
    total_qouta_b = sum(np.ceil(programs['quota']*programs['b_share']/100))
    total_qouta_r = sum(np.ceil(programs['quota']*programs['r_share']/100))
    print(f"Initial matching total b- and r-fills: ({original_total_b:.0f}/{total_qouta_b:.0f}, {original_total_r:.0f}/{total_qouta_r:.0f})")

    # Interestengly, unlike 2022 data, in 2024 data there are two programs (115874 and 168399) of r_only_type (we neglect this and treat them as br-type)

    print("------------------------")
    programs_b_only = programs[programs['b_only_type'] == 1]
    print(f"b-only programs: b-target: {programs_b_only['b_share'].mean()} ({programs_b_only['b_share'].std()})")
    print(f"                 total b-quota: {sum(np.ceil(programs_b_only['quota']*programs_b_only['b_share']/100))}")
    programs_br = programs[programs['br_type'] == 1]
    print(f"br programs: b-target: {programs_br['b_share'].mean()} ({programs_br['b_share'].std()})")
    print(f"             r-target: {programs_br['r_share'].mean()} ({programs_br['r_share'].std()})")
    print(f"             total b-quota: {sum(np.ceil(programs_br['quota']*programs_br['b_share']/100))}")
    print(f"             total r-quota: {sum(np.ceil(programs_br['quota']*programs_br['r_share']/100))}")
    programs_b_only_board = programs_b_only[programs_b_only['internat'] == 1]
    print(f"b-only internats: b-target: {programs_b_only_board['b_share'].mean()} ({programs_b_only_board['b_share'].std()})")
    print(f"                  total b-quota: {sum(np.ceil(programs_b_only_board['quota']*programs_b_only_board['b_share']/100))}")
    programs_b_only_reg = programs_b_only[programs_b_only['internat'] == 0]
    print(f"b-only regular: b-target: {programs_b_only_reg['b_share'].mean()} ({programs_b_only_reg['b_share'].std()})")
    print(f"                total b-quota: {sum(np.ceil(programs_b_only_reg['quota']*programs_b_only_reg['b_share']/100))}")
    programs_br_greater = programs_br[programs_br['b_share'] + programs_br['r_share']>100]
    print(f"br programs (qb+qr>100): b-target: {programs_br_greater['b_share'].mean()} ({programs_br_greater['b_share'].std()})")
    print(f"                         r-target: {programs_br_greater['r_share'].mean()} ({programs_br_greater['r_share'].std()})")
    print(f"                         total b-quota: {sum(np.ceil(programs_br_greater['quota']*programs_br_greater['b_share']/100))}")
    print(f"                         total r-quota: {sum(np.ceil(programs_br_greater['quota']*programs_br_greater['r_share']/100))}")
    programs_br_less = programs_br[programs_br['b_share'] + programs_br['r_share']<=100]
    print(f"br programs (qb+qr<=100): b-target: {programs_br_less['b_share'].mean()} ({programs_br_less['b_share'].std()})")
    print(f"                         r-target: {programs_br_less['r_share'].mean()} ({programs_br_less['r_share'].std()})")
    print(f"                         total b-quota: {sum(np.ceil(programs_br_less['quota']*programs_br_less['b_share']/100))}")
    print(f"                         total r-quota: {sum(np.ceil(programs_br_less['quota']*programs_br_less['r_share']/100))}")
    programs_AA = programs[(programs['b_share'] > 0) | (programs['r_share'] > 0)]
    print(f"AA programs: b-target: {programs_AA['b_share'].mean()} ({programs_AA['b_share'].std()})")
    print(f"             total b-quota: {sum(np.ceil(programs_AA['quota']*programs_AA['b_share']/100))}")

    # Creating size_i-by-size_j matrix for 3d histogram of residentship data:
    #   in each cell (i,j) we put number of students with:
    #       1) share of wishes with residentship in [i*(1/size_i), (i+1)*(1/size_i)) (for last i, we include share 1)
    #       2) number of wishes = j (for last j we also include all students with more wishes)
    residents=resident
    residents['res']=1
    wish_res = wishes.merge(residents, on=['program_id', 'student_id'], how='left')
    wish_res['res'] = wish_res['res'].fillna(0).astype(int)
    wish_counts = wish_res.groupby('student_id').size().rename('wish_num')
    wish_res_counts = wish_res[wish_res['res'] == 1].groupby('student_id').size().rename('wish_res_num')
    stu_res = pd.concat([wish_counts, wish_res_counts], axis = 1)
    stu_res['wish_res_num'] = stu_res['wish_res_num'].fillna(0).astype(int)
    stu_res = stu_res.reset_index()

    size_i = 20 # step = 5%
    size_j = 18
    hist3d_res = np.zeros((size_i,size_j), dtype=int)
    stu_res['res_ratio'] = stu_res['wish_res_num'] / stu_res['wish_num']

    for i in range(size_i):
        lower = i*(1/size_i)
        if i == size_i-1:
            upper = 1.00001
        else:
            upper = (i+1)*(1/size_i)
        ratio_mask = (stu_res['res_ratio'] >= lower) & (stu_res['res_ratio'] < upper)
        for j in range(size_j):
            if j<size_j-1:
                wish_mask = stu_res['wish_num'] == j+1
            else:
                wish_mask = stu_res['wish_num'] > size_j-1

            count = ((ratio_mask) & (wish_mask)).sum()
            hist3d_res[i,j] = count

    df_hist = pd.DataFrame(hist3d_res)
    df_hist.to_csv("hist3d_res_matrix.csv", index=False)
    print("====================================================================")

if use_ROLs:
    if not load_ROLs: # if we want to perform FGH estimation of students' ROLs
        outside_only_students_share = 0.001 # how many candidates who not wished any of sampled programmes to keep in panel
        ROLs_per_student = 30 # number of simulated ROLs per student
        new_keep_set = False # True if we want to sample new subset of programs for estimation

        if new_keep_set:
            STRATAS_FRAC = 0.007 # share of programs for estimation (YY 0.007; NN 0.003; YN 0.003; NY 0.006)
            # Samplimg programs
            keep_set = sampling.sample_programmes(programs, frac = STRATAS_FRAC)

            # SAVING keep_set:
            with open(p/"progs_keep_set.pkl", "wb") as f:
                pickle.dump(keep_set, f)
        else:
            # READING keep_set.pkl:
            with open(p/"progs_keep_set.pkl", "rb") as f:
                keep_set = pickle.load(f)

        # Building long panel
        panel = panel_builder.build_panel_fgh(students, programs, wishes, resident, matches, ranks, keep_set, outside_only_students_share)

        panel = panel[["student_id", "program_id", "chosen", "near_is", "near_sch",
            "bOnly_sch", "br_sch", "br_near", "near_nat", "near_sex", "bShare_sex", "rShare_sex",
            "rShare_sch", "bShare_near", "bShare_sch", "rShare_near", "bOnly_bOnlyWish", "br_brWishRes",
            "logQuota_nat", "logQuota_near", "logQuota_sex", "logQuota_sch",
            "out_sch", "out_sex", "out_nat", "out_wish", "ASC"]].copy()
        panel = panel.fillna(0)
        panel = panel.astype({
            "student_id": "int32",
            "program_id": "int32",
            "chosen": "int8",
            "near_is": "int8",
            "near_sch": "int8",
            "bOnly_sch": "int8",
            "br_sch": "int8",
            "br_near": "int8",
            "near_nat": "int8",
            "near_sex": "int8",
            "bShare_sex": "float",
            "rShare_sex": "float",
            "rShare_sch": "float",
            "bShare_near": "float",
            "bShare_sch": "float",
            "rShare_near": "float",
            "bOnly_bOnlyWish": "float",
            "br_brWishRes": "float",
            "logQuota_nat": "float",
            "logQuota_near": "float",
            "logQuota_sex": "float",
            "logQuota_sch": "float",
            "out_sch": "int8",
            "out_sex": "int8",
            "out_nat": "int8",
            "out_wish": "float",
            "ASC": "int8",
        })
        panel = panel.sort_values(["student_id", "program_id"]).reset_index(drop=True)

        # Estimating MNL: betas and alpha
        del students, programs, wishes, resident, matches, ranks # deleting data for more memory
        gc.collect() # frees up memory
        model = estimate_fgh.estimate(panel)
        params = model.params
        beta_hat = params[~params.index.str.startswith("ASC_")]
        beta_hat = beta_hat.rename_axis("name").reset_index(name="coef")
        beta_hat.to_csv(DATADIR.parent/f"{DATADIR.name}/beta_hat.csv", index = False, float_format="%.6g")

        # OLS-impute alpha_hat for not sampled programs
        # Loading data again
        students, programs, wishes, resident, matches, ranks = load_tables.load_all(DATADIR)

        alpha_hat = alpha_ols.impute_alpha(programs, model)
        alpha_hat.to_csv(DATADIR.parent/f"{DATADIR.name}/alpha_hat.csv", index = False, float_format="%.6g")

        # Estimating s.e. and p-values for stat. significance of model coeffs
        betas = model_statistical_significance.summarize_non_asc_prefix(model)
        print(betas)  # non-ASC coefficients with se, p, and stars

        # Simulate R ROLs per student
        del model # deleting data for more memory
        gc.collect() # frees up memory
        rols = simulate_rols.simulate_ranked_lists(
            R=ROLs_per_student,
            students=students,
            programs=programs,
            wishes=wishes,
            resident=resident,
            alpha_hat=alpha_hat,
            beta_hat=beta_hat,
            out_file=p/"rols.pkl.gz", # or None to keep in memory only
            use_wishes_only=True,
            include_outside_rank=False,
            seed=simulation_seed,
        )
    else: # if all FGH simulation is done
        with gzip.open(p/"rols.pkl.gz", "rb") as f:
            rols = pickle.load(f)

#------------------------
# DA mechanisms
#------------------------

alpha_hat = pd.read_csv(DATADIR.parent/f"{DATADIR.name}/alpha_hat.csv")
beta_hat = pd.read_csv(DATADIR.parent/f"{DATADIR.name}/beta_hat.csv")
SimulationsData2024 = pd.DataFrame(columns=['PCS_matched_studs_amount', 'SF_matched_studs_amount',
                                            'PCS_matched_b_studs_amount', 'SF_matched_b_studs_amount',
                                            'PCS_je_studs_amount', 'SF_je_studs_amount',
                                            'PCS_je_b_studs_amount', 'SF_je_b_studs_amount',
                                            'PCS_95_iter', 'PCS_99_iter', 'PCS_last_iter',
                                            'SF_95_iter', 'SF_99_iter', 'SF_last_iter',
                                            'PCS_avg_b', 'SF_avg_b',
                                            'PCS_avg_r', 'SF_avg_r',
                                            'PCS_total_b', 'SF_total_b',
                                            'PCS_total_r', 'SF_total_r',
                                            'SF_stu_pref', 'PCS_stu_pref',
                                            'SF_b_stu_pref', 'PCS_b_stu_pref',
                                            'SF_only_matched', 'PCS_only_matched',
                                            'SF_only_matched_b', 'PCS_only_matched_b',
                                            'SF_avg_rank_impr', 'PCS_avg_rank_impr',
                                            'SF_avg_rank_impr_b', 'PCS_avg_rank_impr_b'])

for bc in range(bunches_number):
    for mk in range(bunch_size):
        print(f"Market {(bc*bunch_size)+mk+1}/{markets_number} (bunch {bc+1})")
        rols1 = simulate_rols.simulate_ranked_lists(
            R=1,
            students=students,
            programs=programs,
            wishes=wishes,
            resident=resident,
            alpha_hat=alpha_hat,
            beta_hat=beta_hat,
            out_file=None, # or None to keep in memory only
            use_wishes_only=True,
            include_outside_rank=False,
            seed=simulation_seed + (bc * bunch_size) + mk,
        )
        # sampling random profile of students preferences
        prefs = pref_profile_sampling.sample_random_profile(rols1, students)#, seed=42)

        # performing PCS (CDA) and SF-CDA
        PCS_matching, PCS_last_iter, PCS_filled_per_stage = cda_mechanisms.run_cda_fast(programs, students, resident, prefs, ranks, 'call_rank', 'CDA', False)
        SF_matching, SF_last_iter, SF_filled_per_stage = cda_mechanisms.run_cda_fast(programs, students, resident, prefs, ranks, 'merit_rank', 'SF-CDA', False)
        PCS_matched_studs_amount = len(PCS_matching)
        SF_matched_studs_amount = len(SF_matching)

        stu_b = students[students['b_trait'] == 1]['student_id']
        PCS_matched_b_studs = PCS_matching[PCS_matching['student_id'].isin(stu_b)]
        PCS_matched_b_studs_amount = len(PCS_matched_b_studs)
        SF_matched_b_studs = SF_matching[SF_matching['student_id'].isin(stu_b)]
        SF_matched_b_studs_amount = len(SF_matched_b_studs)

        # Calculating 95%-fill and 99%-fill stages for both mechs
        PCS_95_iter = next(i for i, val in enumerate(PCS_filled_per_stage) if val >= 0.95 * PCS_filled_per_stage[-1])
        PCS_99_iter = next(i for i, val in enumerate(PCS_filled_per_stage) if val >= 0.99 * PCS_filled_per_stage[-1])
        SF_95_iter = next(i for i, val in enumerate(SF_filled_per_stage) if val >= 0.95 * SF_filled_per_stage[-1])
        SF_99_iter = next(i for i, val in enumerate(SF_filled_per_stage) if val >= 0.99 * SF_filled_per_stage[-1])

        # Amount of students with je for both mechs
        PCS_je_studs_amount, PCS_je_b_studs_amount = cda_mechanisms.number_of_students_with_je(PCS_matching, prefs, ranks, programs, students, resident)
        SF_je_studs_amount, SF_je_b_studs_amount = cda_mechanisms.number_of_students_with_je(SF_matching, prefs, ranks, programs, students, resident)

        # Percentage of students with je across mechanisms
        stu_n = students['student_id'].nunique()
        PCS_je_students_perc = (PCS_je_studs_amount/stu_n) * 100
        SF_je_students_perc = (SF_je_studs_amount/stu_n) * 100

        # Average b and r compliances
        PCS_avg_b, PCS_avg_r, PCS_total_b, PCS_total_r = cda_mechanisms.avg_traits_compl(PCS_matching, programs, students, resident)
        SF_avg_b, SF_avg_r, SF_total_b, SF_total_r = cda_mechanisms.avg_traits_compl(SF_matching, programs, students, resident)

        # PE comparison
        SF_stu_pref, PCS_stu_pref, SF_b_stu_pref, PCS_b_stu_pref, SF_only_matched, PCS_only_matched, SF_only_matched_b, PCS_only_matched_b, SF_avg_rank_impr, PCS_avg_rank_impr, SF_avg_rank_impr_b, PCS_avg_rank_impr_b = two_matchings_PE_comp.compare_matchings_efficiency(SF_matching, PCS_matching, prefs, students)

        # Combining results to df
        res = [PCS_matched_studs_amount, SF_matched_studs_amount,
               PCS_matched_b_studs_amount, SF_matched_b_studs_amount,
                PCS_je_studs_amount, SF_je_studs_amount,
                PCS_je_b_studs_amount, SF_je_b_studs_amount,
                PCS_95_iter, PCS_99_iter, PCS_last_iter,
                SF_95_iter, SF_99_iter, SF_last_iter,
                PCS_avg_b, SF_avg_b,
                PCS_avg_r, SF_avg_r,
                PCS_total_b, SF_total_b,
                PCS_total_r, SF_total_r,
                SF_stu_pref, PCS_stu_pref,
                SF_b_stu_pref, PCS_b_stu_pref,
                SF_only_matched, PCS_only_matched,
                SF_only_matched_b, PCS_only_matched_b,
                SF_avg_rank_impr, PCS_avg_rank_impr,
                SF_avg_rank_impr_b, PCS_avg_rank_impr_b]
        SimulationsData2024.loc[len(SimulationsData2024)] = res

        #print(SimulationsData2024.iloc[[mk]])

    # Saving resulting data for current bunch
    filename = f"SimulationsData{result_year}_{bc}.csv"
    SimulationsData2024.to_csv(p/filename, index=False)
    print(f"Bunch {bc+1} with {bunch_size} markets ({bc*bunch_size+1}:{(bc+1)*bunch_size}) is saved!")

filename = f"SimulationsData{result_year}_{bc}.csv"
SimulationsData2024 = pd.read_csv(p/filename)

if final_analysis:
    # Adding column with SF - PCS differences and checking whether their means are different from 0
    from scipy.stats import ttest_1samp
    SimulationsData2024['NET_matched_studs_amount'] = SimulationsData2024['SF_matched_studs_amount'] - SimulationsData2024['PCS_matched_studs_amount']
    sample = SimulationsData2024['NET_matched_studs_amount'].dropna()
    t_stat, p_matched_studs_amount = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_matched_b_studs_amount'] = SimulationsData2024['SF_matched_b_studs_amount'] - SimulationsData2024['PCS_matched_b_studs_amount']
    sample = SimulationsData2024['NET_matched_b_studs_amount'].dropna()
    t_stat, p_matched_b_studs_amount = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_je_studs_amount'] = SimulationsData2024['SF_je_studs_amount'] - SimulationsData2024['PCS_je_studs_amount']
    sample = SimulationsData2024['NET_je_studs_amount'].dropna()
    t_stat, p_je_studs_amount = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_je_b_studs_amount'] = SimulationsData2024['SF_je_b_studs_amount'] - SimulationsData2024['PCS_je_b_studs_amount']
    sample = SimulationsData2024['NET_je_b_studs_amount'].dropna()
    t_stat, p_je_b_studs_amount = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_last_iter'] = SimulationsData2024['SF_last_iter'] - SimulationsData2024['PCS_last_iter']
    sample = SimulationsData2024['NET_last_iter'].dropna()
    t_stat, p_last_iter = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_99_iter'] = SimulationsData2024['SF_99_iter'] - SimulationsData2024['PCS_99_iter']
    sample = SimulationsData2024['NET_99_iter'].dropna()
    t_stat, p_99_iter = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_95_iter'] = SimulationsData2024['SF_95_iter'] - SimulationsData2024['PCS_95_iter']
    sample = SimulationsData2024['NET_95_iter'].dropna()
    t_stat, p_95_iter = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_avg_b'] = SimulationsData2024['SF_avg_b'] - SimulationsData2024['PCS_avg_b']
    sample = SimulationsData2024['NET_avg_b'].dropna()
    t_stat, p_avg_b = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_avg_r'] = SimulationsData2024['SF_avg_r'] - SimulationsData2024['PCS_avg_r']
    sample = SimulationsData2024['NET_avg_r'].dropna()
    t_stat, p_avg_r = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_total_b'] = SimulationsData2024['SF_total_b'] - SimulationsData2024['PCS_total_b']
    sample = SimulationsData2024['NET_total_b'].dropna()
    t_stat, p_total_b = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_total_r'] = SimulationsData2024['SF_total_r'] - SimulationsData2024['PCS_total_r']
    sample = SimulationsData2024['NET_total_r'].dropna()
    t_stat, p_total_r = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_stu_pref'] = SimulationsData2024['SF_stu_pref'] - SimulationsData2024['PCS_stu_pref']
    sample = SimulationsData2024['NET_stu_pref'].dropna()
    t_stat, p_stu_pref = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_b_stu_pref'] = SimulationsData2024['SF_b_stu_pref'] - SimulationsData2024['PCS_b_stu_pref']
    sample = SimulationsData2024['NET_b_stu_pref'].dropna()
    t_stat, p_b_stu_pref = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_only_matched'] = SimulationsData2024['SF_only_matched'] - SimulationsData2024['PCS_only_matched']
    sample = SimulationsData2024['NET_only_matched'].dropna()
    t_stat, p_only_matched = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_only_matched_b'] = SimulationsData2024['SF_only_matched_b'] - SimulationsData2024['PCS_only_matched_b']
    sample = SimulationsData2024['NET_only_matched_b'].dropna()
    t_stat, p_only_matched_b = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_avg_rank_impr'] = SimulationsData2024['SF_avg_rank_impr'] - SimulationsData2024['PCS_avg_rank_impr']
    sample = SimulationsData2024['NET_avg_rank_impr'].dropna()
    t_stat, p_avg_rank_impr = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    SimulationsData2024['NET_avg_rank_impr_b'] = SimulationsData2024['SF_avg_rank_impr_b'] - SimulationsData2024['PCS_avg_rank_impr_b']
    sample = SimulationsData2024['NET_avg_rank_impr_b'].dropna()
    t_stat, p_avg_rank_impr_b = ttest_1samp(sample, popmean = 0) # if p < 0.05 -> not 0

    # Reordering columns
    SimulationsData2024 = SimulationsData2024[['PCS_avg_b', 'SF_avg_b', 'NET_avg_b',
                                            'PCS_avg_r', 'SF_avg_r', 'NET_avg_r',
                                            'PCS_total_b', 'SF_total_b', 'NET_total_b',
                                            'PCS_total_r', 'SF_total_r', 'NET_total_r',
                                            'PCS_je_studs_amount', 'SF_je_studs_amount', 'NET_je_studs_amount',
                                            'PCS_je_b_studs_amount', 'SF_je_b_studs_amount', 'NET_je_b_studs_amount',
                                            'PCS_matched_studs_amount', 'SF_matched_studs_amount', 'NET_matched_studs_amount',
                                            'PCS_matched_b_studs_amount', 'SF_matched_b_studs_amount', 'NET_matched_b_studs_amount',
                                            'PCS_last_iter', 'SF_last_iter', 'NET_last_iter',
                                            'PCS_99_iter', 'SF_99_iter', 'NET_99_iter',
                                            'PCS_95_iter', 'SF_95_iter', 'NET_95_iter',
                                            'PCS_stu_pref', 'SF_stu_pref', 'NET_stu_pref',
                                            'PCS_b_stu_pref', 'SF_b_stu_pref', 'NET_b_stu_pref',
                                            'PCS_only_matched', 'SF_only_matched', 'NET_only_matched',
                                            'PCS_only_matched_b', 'SF_only_matched_b', 'NET_only_matched_b',
                                            'PCS_avg_rank_impr', 'SF_avg_rank_impr', 'NET_avg_rank_impr',
                                            'PCS_avg_rank_impr_b', 'SF_avg_rank_impr_b', 'NET_avg_rank_impr_b']]


    summary = pd.DataFrame({
        'Mean': SimulationsData2024.mean(),
        'Std': SimulationsData2024.std()
    })

    print(summary)
