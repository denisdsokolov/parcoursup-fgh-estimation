import pandas as pd
import numpy as np
import pickle
import gc
from datetime import datetime
import warnings
from pathlib import Path
from project_config import env_bool, env_int, env_path

year = env_int("PARCOURSUP_YEAR", 24, minimum=0)

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

DATADIR = env_path("PARCOURSUP_DATA_DIR", "FGH_data" + "_" + tag_c + "_" + tag_q) # folder to save data
DATADIR.mkdir(parents=True, exist_ok=True)

#region PCS year DATA PREPROCESSING: TABLES READING

#======================================================================================================
#======================================================================================================
"""G_CAN table: information on candidates
# G_CN_COD: candidate id
# G_CN_SEX_FEM: 1 - female, 0 - male
# G_CN_BRS: 0 - non boursier, 1 - secondary school b., 2 - higher education b. (1 => boursier trait b)
# G_CN_FLG_RA: 1 - candidate activated automatic responder
# G_CN_NAT: F - french, E - other
# G_CN_COD_POS: Postal code
"""
read_g_can = False # g_can is loaded from preprocessed pkl file

if read_g_can:
    name = f"g_can{year}.csv"
    g_can_raw = pd.read_csv(name,encoding = "ISO-8859-1")
    # print(g_can.dtypes)
    #PROBLEM: csv file contains errors: text got inserted into the first column ('G_TI_COD')
    g_can2 = g_can_raw[g_can_raw['G_CN_COD'].map(g_can_raw['G_CN_COD'].value_counts()) == 1] #PROBLEM SOLVED PARTIALLY: leave only rows with unique entry in the column 'G_CN_COD', now # of rows is still more than in the initial STATA file g_can'
    numeric_G_CN_COD = pd.to_numeric(g_can2['G_CN_COD'], errors = 'coerce')
    numeric_G_CN_COD_clean = numeric_G_CN_COD[~np.isnan(numeric_G_CN_COD) & (numeric_G_CN_COD != np.inf)]
    g_can = g_can2.loc[numeric_G_CN_COD_clean.index[numeric_G_CN_COD_clean % 1 == 0]] #PROBLEM SOLVED PARTIALLY but better: leaving only rows where 'G_CN_COD' is int: for some reason still contains 10 more rows???
    g_can = g_can.reset_index(drop=True)
    g_can['G_CN_COD'] = g_can['G_CN_COD'].astype(int)

    # SAVING g_can:
    with open("g_can.pkl", "wb") as f:
        pickle.dump(g_can, f)

# READING g_can.pkl:
with open("g_can.pkl", "rb") as f:
    g_can = pickle.load(f)

#======================================================================================================
#======================================================================================================
"""G_TRI_INS table: information on institutions (G_TI_COD)
# G_TI_COD: id for registration training
# G_FR_COD_INS: id for the type of formation (consult table G_FOR) -- useful for building stratas for groups (42 and 43 are too big - can use G_FL_COD_INS for them to make them more detailed)
# G_FL_COD_INS: id for the sub-type of formation (consult table G_FIL)
"""
read_g_tri_ins = False # if false - g_tri_ins is loaded from preprocessed pkl file

if read_g_tri_ins:
    name = f"g_tri_ins{year}.csv"
    g_tri_ins_raw = pd.read_csv(name,encoding = "ISO-8859-1")
    # print(g_tri_ins.dtypes)
    #PROBLEM: csv file contains errors: text got inserted into the first column ('G_TI_COD')
    g_tri_ins2 = g_tri_ins_raw[g_tri_ins_raw['G_TI_COD'].map(g_tri_ins_raw['G_TI_COD'].value_counts()) == 1] #PROBLEM SOLVED PARTIALLY: leave only rows with unique entry in the column 'G_TI_COD', now # of rows is still more than in the initial STATA file 'g_tri_ins' (22,071)
    numeric_G_TI_COD = pd.to_numeric(g_tri_ins2['G_TI_COD'], errors = 'coerce')
    numeric_G_TI_COD_clean = numeric_G_TI_COD[~np.isnan(numeric_G_TI_COD) & (numeric_G_TI_COD != np.inf)]
    g_tri_ins = g_tri_ins2.loc[numeric_G_TI_COD_clean.index[numeric_G_TI_COD_clean % 1 == 0]] #PROBLEM SOLVED!
    g_tri_ins = g_tri_ins.reset_index(drop=True)
    g_tri_ins['G_TI_COD'] = g_tri_ins['G_TI_COD'].astype(int)

    # SAVING g_tri_ins:
    with open("g_tri_ins.pkl", "wb") as f:
        pickle.dump(g_tri_ins, f)

# READING g_tri_ins.pkl:
with open("g_tri_ins.pkl", "rb") as f:
    g_tri_ins = pickle.load(f)

#======================================================================================================
#======================================================================================================
"""A_REC table: list of available courses
# G_TI_COD: id for registration training
# G_TA_COD: id for assignment training (one G_TI_COD may be assigned to many G_TA_COD, but not vice-versa)
# A_RC_CAP: total quota (NOT FOR GROUP)
# A_RC_CAP_INT_GAR: boarding school quota for boys
# A_RC_CAP_INT_FIL: boarding school quota for girls
# A_RC_CAP_INT_MIX: boarding school quota for mixed dorm
# A_RC_TAU_BRS_REC: min target share of boursiers (out of 100%)
# A_RC_FLG_TAU_BRS: 1 - if has target share for b, 0 - otw
# A_RC_TAU_NON_RES_REC: max target share of non-residents (out of 100%)
# A_RC_FLG_TAU_NON_RES: 1 - if has target share for r, 0 - otw
"""

name = f"a_rec{year}.csv"
a_rec_raw = pd.read_csv(name,encoding = "ISO-8859-1")
#PROBLEM2: csv file contains errors: text got inserted into the first column ('G_TI_COD')
a_rec = a_rec_raw[a_rec_raw['G_TA_COD'].map(a_rec_raw['G_TA_COD'].value_counts()) == 1] #PROBLEM2 SOLVED: leave only rows with unique entry in the column 'G_TA_COD', now # of rows is the same as in the initial STATA file 'a_rec'

# All internats:
internats_all =  a_rec[(a_rec['A_RC_CAP_INT_GAR'].notna()) | (a_rec['A_RC_CAP_INT_FIL'].notna()) | (a_rec['A_RC_CAP_INT_MIX'].notna())] #619 internats (formations with some dorm quota)

#======================================================================================================
#======================================================================================================
"""A_REC_GRP table: list of available groups
# C_GP_COD: id for group (program) (one G_TA_COD may be assigned to many C_GP_COD AND vice-versa)
# G_TI_COD: id for registration training
# G_TA_COD: id for assignment training (one G_TI_COD may be assigned to many G_TA_COD, but not vice-versa)
# A_RG_PLA: total quota of a group
"""
name = f"a_rec_grp{year}.csv"
a_rec_grp = pd.read_csv(name,encoding = "ISO-8859-1") # contains group-institution corresp. and quotas of groups

# Each row of 'a_rec_grp' per one group?
number_of_institutions_group_data = a_rec_grp['G_TA_COD'].nunique() # each initial institution has at least one group
number_of_groups = a_rec_grp['C_GP_COD'].nunique() # more groups than rows, maybe need to remove all groups with no seat quota:
a_rec_grp_no_nan_quota = a_rec_grp.dropna(subset = ['A_RG_PLA']) # dropped all rows without seat quota
a_rec_grp_no_nan_quota = a_rec_grp_no_nan_quota[a_rec_grp_no_nan_quota['A_RG_PLA'] != 0] # dropped all rows with 0 seat quota
number_of_groups_no_nan_quota = a_rec_grp_no_nan_quota['C_GP_COD'].nunique() # still more rows of data than groups
a_rec_grp_processed = a_rec_grp_no_nan_quota[~a_rec_grp_no_nan_quota['C_GP_COD'].duplicated(keep=False)] # removing all rows where the same group meets more than once
# NOW: a_rec_grp_processed contains one unique group per row (no two rows contain the same group C_GP_COD)

#======================================================================================================
#======================================================================================================
"""C_CAN_GRP table: table with merit rankings and call orders
# G_CN_COD: candidate id
# C_GP_COD: id for group (program) (one G_TA_COD may be assigned to many C_GP_COD, but not vice-versa)
# C_CG_RAN: merit rank of a candidate for a program
# C_CG_ORD_APP: call order rank of a candidate for a program
"""
name = f"c_can_grp{year}.csv"
c_can_grp = pd.read_csv(name,encoding = "ISO-8859-1") # contains ranking and call order for each group

c_can_grp_processed = c_can_grp.dropna(subset = ['C_CG_RAN']) # dropped all rows without initial ranks of candidates

#======================================================================================================
#======================================================================================================
"""I_INS table: table with resident trait data
# G_CN_COD: candidate id
# G_TI_COD: id for registration training
# I_IS_FLC_SEC: binary, indicates whether a candidate G_CN_COD is a resident for an institution G_TI_COD
"""
read_i_ins = False # i_ins is loaded from preprocessed pkl file

if read_i_ins:
    name = f"i_ins{year}.csv"
    i_ins = pd.read_csv(name,encoding = "ISO-8859-1") # should be 11 167 167 rows

    # SAVING i_ins:
    with open("i_ins.pkl", "wb") as f:
        pickle.dump(i_ins, f)

# READING i_ins.pkl:
with open("i_ins.pkl", "rb") as f:
    i_ins = pickle.load(f)

#======================================================================================================
#======================================================================================================
"""A_VOE table: list with wishes
# G_CN_COD: candidate id
# G_TA_COD: id for assignment training (one G_TI_COD may be assigned to many G_TA_COD, but not vice-versa)
# I_RH_COD: 1 - wish with a room, 0 - without (boarding programs)
# A_SV_COD: respond to proposition with this wish: 5,6,10 - yes; -31,-93,-50,-51,-53,-40,-41,-42,-30,-34,-25,-20,-21,-23,-24,-22,-43,-35,-36,-26,-18,-19,-27,-29,-28,-33,-94,-32,-39 - no
# A_VE_ORD: rank of this wish if applicant activated automatic responce
"""

read_a_voe = True # a_voe is loaded from preprocessed pkl file

if read_a_voe:
    name = f"a_voe{year}.csv"
    a_voe = pd.read_csv(name,encoding = "ISO-8859-1") # should be 21 006 901 rows

    # SAVING a_voe:
    with open("a_voe.pkl", "wb") as f:
        pickle.dump(a_voe, f)

# READING a_voe.pkl:
with open("a_voe.pkl", "rb") as f:
    a_voe = pickle.load(f)

#======================================================================================================
#======================================================================================================
"""SP_G_TRI_AFF table: locations of formations (for stratas)
# G_TA_COD: id for assignment training (one G_TI_COD may be assigned to many G_TA_COD, but not vice-versa)
# G_EA_COD_POS: postal code for a formation (for stratas)
"""

read_sp_g_tri_aff = False # if false - sp_g_tri_aff is loaded from preprocessed pkl file

if read_sp_g_tri_aff:
    name = f"sp_g_tri_aff{year}.csv"
    sp_g_tri_aff_raw = pd.read_csv(name,encoding = "ISO-8859-1") # should be 22,924 rows
    # print(sp_g_tri_aff.dtypes)
    #PROBLEM: csv file contains errors: text got inserted into the first column ('G_TA_COD')
    sp_g_tri_aff2 = sp_g_tri_aff_raw[sp_g_tri_aff_raw['G_TA_COD'].map(sp_g_tri_aff_raw['G_TA_COD'].value_counts()) == 1] #PROBLEM SOLVED PARTIALLY: leave only rows with unique entry in the column 'G_TA_COD', now # of rows is still more than in the initial STATA file 'g_tri_ins' (22,071)
    numeric_G_TA_COD = pd.to_numeric(sp_g_tri_aff2['G_TA_COD'], errors = 'coerce')
    numeric_G_TA_COD_clean = numeric_G_TA_COD[~np.isnan(numeric_G_TA_COD) & (numeric_G_TA_COD != np.inf)]
    sp_g_tri_aff = sp_g_tri_aff2.loc[numeric_G_TA_COD_clean.index[numeric_G_TA_COD_clean % 1 == 0]] #PROBLEM SOLVED!
    sp_g_tri_aff = sp_g_tri_aff.reset_index(drop=True)
    sp_g_tri_aff['G_TA_COD'] = sp_g_tri_aff['G_TA_COD'].astype(int)

    # SAVING sp_g_tri_aff:
    with open("sp_g_tri_aff.pkl", "wb") as f:
        pickle.dump(sp_g_tri_aff, f)

# READING sp_g_tri_aff.pkl:
with open("sp_g_tri_aff.pkl", "rb") as f:
    sp_g_tri_aff = pickle.load(f)

#======================================================================================================
#======================================================================================================
"""A_ADM table: list of all admission propositions
# G_CN_COD: candidate id
# G_TI_COD: id for registration training
# G_TA_COD: id for assignment training (one G_TI_COD may be assigned to many G_TA_COD, but not vice-versa)
# C_GP_COD: id for group (program)
# A_TA_COD: 1 - main admission phase proposition, 3 - additional admission phase
# A_SV_COD: respond to proposition: 5,6,10 - yes; -31,-93,-50,-51,-53,-40,-41,-42,-30,-34,-25,-20,-21,-23,-24,-22,-43,-35,-36,-26,-18,-19,-27,-29,-28,-33,-94,-32,-39 - no
# I_RH_COD: 1 - proposition with a room, 0 - without (boarding programs)
"""
name = f"a_adm{year}.csv"
a_adm = pd.read_csv(name,encoding = "ISO-8859-1")

# =================== Looking for propositions with a room made not by internats with dorm quotas (internats_all): ==========================

#How many propositions were made by internats (propositions with [a_adm['I_RH_COD'] == 1] should have been made only by internats)
df_propositions_by_internats = a_adm[a_adm['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 254111
# how many of them were made with a room?
num_of_propositions_by_internats_with_room = (df_propositions_by_internats['I_RH_COD'] == 1).sum() # 44606 made by internats
num_of_propositions_with_room = (a_adm['I_RH_COD'] == 1).sum() # 60323 made in total (15k were made by whom?????)
# lets find institutions which made a proposition with a room and do not have a dorm quota in the data
propositions_by_internats_with_room = df_propositions_by_internats[df_propositions_by_internats['I_RH_COD'] == 1]
propositions_with_room = a_adm[a_adm['I_RH_COD'] == 1]

# Find all 'G_TA_COD', s.t. it made a proposition with a room but is not in my list of internats
propositions_by_NOT_internats_with_room = propositions_with_room[~propositions_with_room['G_TA_COD'].isin(propositions_by_internats_with_room['G_TA_COD'])] # exactly 15717 left propositions
# TA of the first entry is 2112, here are it quotas from a_rec (should be no quotas)
institution_with_G_TA_COD_2112_a_rec_row = a_rec[a_rec['G_TA_COD'] == 2112]
#print(institution_with_G_TA_COD_2112_a_rec_row[['A_RC_CAP_INT_GAR','A_RC_CAP_INT_FIL','A_RC_CAP_INT_MIX']])
# So, for instance, for institution with G_TA_COD = 2112 we dont have info on its dorm quotas!
# So, lets create a list of internats without data for dorm quotas (which proposed with a room):
institutions_proposed_room = a_rec[a_rec['G_TA_COD'].isin(propositions_with_room['G_TA_COD'])] # all institutions that proposed room at least once
institutions_proposed_room_and_have_data_on_dorm_quota = institutions_proposed_room[institutions_proposed_room['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 614 out of 619 internats with data on dorm quotas made at leats one proposition with a room (all 619 proposed though)
institutions_proposed_room_and_dont_have_data_on_dorm_quota = institutions_proposed_room[~institutions_proposed_room['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 138 (potential) internats dont have data on dorm quotas: BAD_DATA
# RESULT:
internats_no_dorm_quota = institutions_proposed_room_and_dont_have_data_on_dorm_quota

# =================== Need to have propositions from only main admission phase: ===================
#a_adm_main = a_adm[(a_adm['A_TA_COD'] == 1) & (~a_adm['G_TA_COD'].isin(internats_all['G_TA_COD'])) & (~a_adm['G_TA_COD'].isin(internats_no_dorm_quota['G_TA_COD']))] # if want to exclude internats
a_adm_main = a_adm[a_adm['A_TA_COD'] == 1] # keeping only main admission phase propositions
#endregion

#====================================================================================================
#====================================DATA PROCESSING=================================================
#====================================================================================================

#region Working with propositions: extracting all groups and candidates with the final matching of the main phase
df_propositions = a_adm_main.copy()  # keep original clean

# For each responce calculating when it was done relative to the first day:
START = pd.Timestamp("2022-01-01")  # reference day (midnight) Jan 1st 2022

df_propositions['A_AM_DAT_REP_FORMATTED'] = pd.to_datetime(df_propositions['A_AM_DAT_REP'], format='%d%b%Y:%H:%M:%S')
df_propositions['response_day'] = (df_propositions['A_AM_DAT_REP_FORMATTED'] - START).dt.days.astype("Int32")
df_propositions['response_day'] = (df_propositions['response_day'] - df_propositions['response_day'].min() + 1) # the first day is now 1

# Changing responses IDs to 1 and -1 - accept/refuse:
df_propositions['response'] = np.sign(df_propositions['A_SV_COD']).astype(int)

# Renaming columns:
rename_map = {
    'G_CN_COD': 'student_id',
    'G_TI_COD': 'TI_id',
    'G_TA_COD': 'TA_id',
    'C_GP_COD': 'program_id'
}
df_propositions = df_propositions.rename(columns=rename_map)

# For df_propositions table we leave only needed info:
cols_to_keep = ['student_id', 'TI_id', 'TA_id', 'program_id', 'response', 'response_day']
df_propositions = df_propositions.loc[:, cols_to_keep] # processed propositions table
#df_propositions.isna().any().any() # checks that no missings

# Searching for the latest 'yes' for each student:
accepts = df_propositions[df_propositions['response'] == 1] # keeping only 'yes'
idx_max_day = accepts.groupby('student_id')['response_day'].idxmax() # for each student finding index of the row with maximal day
final_prog = accepts.loc[idx_max_day, ['student_id', 'program_id', 'response_day']].reset_index(drop=True) # list of pairs (student,last accepted program, day of acceptance)

# Checking that any student did not reject her last 'yes' later:
pairs = final_prog[['student_id', 'program_id']].drop_duplicates() # no dublicates
final_prog_responses = (
    df_propositions.merge(pairs, on=['student_id', 'program_id'], how='inner').sort_values(['student_id', 'program_id', 'response_day'])
) # all responses of a student to her final accepted program
idx_latest_resp = (
    final_prog_responses.groupby(['student_id', 'program_id'])['response_day'].idxmax()
) # idexes of rows with the max response day fro each student
final_prog_last_responses = final_prog_responses.loc[idx_latest_resp].reset_index(drop=True) # last responses to each student's final accepted program

# if the response is -1 and response day is larger => she went to outside option
baseline = (
    final_prog[['student_id', 'program_id', 'response_day']].rename(columns={'response_day': 'baseline_day'})
) # for each pair of a student and a last accepted program we write 'baseline_day' as the day of acceptance

final_prog_latest_response = final_prog_last_responses.merge(baseline, on=['student_id', 'program_id'], how='left') # merging two tables
final_prog_latest_response = final_prog_latest_response[final_prog_latest_response['response_day'] > final_prog_latest_response['baseline_day']] # leaving only pairs of a student and latest accepted program that later was rejected

cols_to_keep = ['student_id', 'program_id']
df_final_prog_rejected = final_prog_latest_response.loc[:, cols_to_keep] # table with pairs of a student and latest accepted program that later rejected this program => these students are unmatched

# ================================= Constructing the final MATCHING =========================================
cols_to_keep = ['student_id', 'program_id']
df_matching = final_prog.loc[:, cols_to_keep] # all pairs of a student and latest accepted program

if excl_cand: # if we do not treat students that rejected last accepted program as matched to this program
    rejected_idx = df_final_prog_rejected.set_index(['student_id', 'program_id']).index
    keep_mask = ~df_matching.set_index(['student_id', 'program_id']).index.isin(rejected_idx)
    df_matching = df_matching[keep_mask].reset_index(drop=True) # all pairs of a student and latest accepted program that was not rejected later

# ================================= Data for all ranks of participated programs (groups) =============================
# Renaming columns:
rename_map = {
    'G_CN_COD': 'student_id',
    'C_GP_COD': 'program_id',
    'C_CG_RAN': 'merit_rank',
    'C_CG_ORD_APP': 'call_rank'
}
df_ranks = c_can_grp_processed.rename(columns=rename_map)
cols_to_keep = ['student_id', 'program_id', 'merit_rank', 'call_rank']
df_ranks = df_ranks[cols_to_keep]
progs_proposed = df_propositions['program_id'].unique().tolist() # list with all participated (proposed during the main admission stage) programs
df_ranks = (
    df_ranks[df_ranks['program_id'].isin(progs_proposed)].copy().reset_index(drop=True)
) # leaving only proposed programs' data

# SANITY CHECK: whether each pair of a student and a program from df_matching appears in df_ranks:
matching_idx = df_matching.set_index(['student_id', 'program_id']).index
ranks_idx = df_ranks.set_index(['student_id', 'program_id']).index
mask = matching_idx.isin(ranks_idx)
df_matching['in_ranks'] = mask # False if not - MANY Falses
df_matching = df_matching[df_matching['in_ranks'] == True] # FINAL MATCHING (excluded all matched student-program pairs without initial ranking)
df_matching = df_matching.drop(columns=['in_ranks']).drop_duplicates()

# ================================= Data for all participated programs (groups) =============================
df_programs = (
    df_propositions[df_propositions['program_id'].isin(progs_proposed)].copy()
) # all programs that proposed during the main stage
df_programs.reset_index(drop=True, inplace=True) # reseting index
cols_to_keep = ['TI_id', 'TA_id', 'program_id']
df_programs = df_programs.loc[:, cols_to_keep] # keeping only IDs
df_programs = df_programs.drop_duplicates()
df_programs.reset_index(drop=True, inplace=True) # reseting index

df_internats = (
    pd.concat(
        [internats_all[['G_TA_COD', 'G_TI_COD']],
         internats_no_dorm_quota[['G_TA_COD', 'G_TI_COD']]],
        ignore_index=True
    ).drop_duplicates()
)

# Renaming columns:
rename_map = {
    'G_TI_COD': 'TI_id',
    'G_TA_COD': 'TA_id'
}
df_internats = df_internats.rename(columns=rename_map)

# -------------------- Adding binary 'is internat' column:

mask = df_programs['TA_id'].isin(df_internats['TA_id'])
df_programs['internat'] = mask.astype(int) # binary column 'internat' contains 1 for each internat program

# -------------------- Adding quotas column:
quota_map = (
    a_rec_grp_processed.set_index('C_GP_COD')['A_RG_PLA']
)
df_programs['quota_data'] = df_programs['program_id'].map(quota_map) # filling program quotas from the data - many NaNs
df_programs.fillna(0, inplace=True) # all NaN quotas -> 0

# For all programs: see how many students matched and adjust quotas:
matched_per_program = df_matching['program_id'].value_counts() # how many students are matched per each program
df_programs['amount_matched'] = (
    df_programs['program_id'].map(matched_per_program).fillna(0).astype(int)
)
df_programs['quota'] = df_programs[['quota_data', 'amount_matched']].max(axis=1).astype(int) # adjusted quotas (max among # matched and data quota)

# QUESTIONABLE QUOTA CHANGE:
if excl_seats:
    # Another sanity correction: no unmatched student should have wished a program with empty seats under the final matching
    unmatched_students = df_ranks.loc[~df_ranks['student_id'].isin(df_matching['student_id']), 'student_id'].drop_duplicates() # all unmatched students
    unmatched_students_wishes = df_ranks.loc[df_ranks['student_id'].isin(unmatched_students), 'program_id'].drop_duplicates() # all wishes of unmatched students

    not_filled_progs = df_programs.loc[df_programs['quota'] > df_programs['amount_matched'], 'program_id'] # all programs with empty seats
    not_filled_progs_are_unmatched_students_wishes = not_filled_progs[not_filled_progs.isin(unmatched_students_wishes)].unique() # all programs with empty seat that were wished by some unmatched student

    mask = df_programs['program_id'].isin(not_filled_progs_are_unmatched_students_wishes)
    df_programs.loc[mask, 'quota'] = np.minimum(
        df_programs.loc[mask, 'quota'],
        df_programs.loc[mask, 'amount_matched']
    ) # for each such program make its quota exactly number of matched students

# -------------------- Adding b-share column:
b_share_map = (
    a_rec.set_index('G_TA_COD')['A_RC_TAU_BRS_REC']
)
df_programs['b_share'] = df_programs['TA_id'].map(b_share_map)
df_programs['b_share'] = df_programs['b_share'].fillna(0)

# -------------------- Adding r-share column:
r_share_map = (
    a_rec.set_index('G_TA_COD')['A_RC_TAU_NON_RES_REC']
)
df_programs['r_share'] = df_programs['TA_id'].map(r_share_map)
df_programs['r_share'] = 100 - df_programs['r_share']
df_programs['r_share'] = df_programs['r_share'].fillna(0)

# -------------------- Adding binary 'b-only-type' column:
df_programs['b_only_type'] = ((df_programs['b_share'] > 0) & (df_programs['r_share'] == 0)).astype(int)

# -------------------- Adding binary 'br-type' column:
df_programs['br_type'] = ((df_programs['b_share'] > 0) & (df_programs['r_share'] > 0)).astype(int)

# -------------------- Adding binary 'no-AA-type' column:
df_programs['no_aa_type'] = ((df_programs['b_share'] == 0) & (df_programs['r_share'] == 0)).astype(int)

# -------------------- Adding column with kinds:
kind_map = (
    g_tri_ins.set_index('G_TI_COD')['G_FR_COD_INS']
)
df_programs['kind'] = df_programs['TI_id'].map(kind_map)

# -------------------- Adding column with post codes:
post_map = (
    sp_g_tri_aff.set_index('G_TA_COD')['G_EA_COD_POS']
)
df_programs['post_code'] = df_programs['TA_id'].map(post_map)
df_programs['post_code'] = df_programs['post_code'].fillna(0) # 15 programs dont have post codes

# -------------------- FINAL TABLE with programs data:
cols_to_keep = ['TI_id', 'TA_id', 'program_id', 'quota', 'b_share', 'r_share', 'b_only_type',
                'br_type', 'no_aa_type', 'kind', 'post_code', 'internat']
df_programs = df_programs[cols_to_keep] # one 'program_id'may have many 'TA_id'
df_programs_neat = df_programs.drop(columns=['TI_id', 'TA_id', 'post_code']).drop_duplicates() # has only 'program_id' => no post codes

#------------------------ Adding 'wished_by' column to df_programs_neat: #(students wished this program):
app_counts = df_ranks['program_id'].value_counts()
df_programs_neat['wished_by'] = (
    df_programs_neat['program_id'].map(app_counts).fillna(0).astype(int)
)
# Some programs are not ranked by anyone: adding column 'amount_matched':
matched_per_program = df_matching['program_id'].value_counts() # how many students are matched per each program
df_programs_neat['amount_matched'] = (
    df_programs_neat['program_id'].map(matched_per_program).fillna(0).astype(int)
)
# Excluding all programs with no students who wished it (there are some programs who admitted some students but never ranked them, this is BAD DATA):
df_programs_neat = df_programs_neat[df_programs_neat['wished_by'] > 0]
# Excluding all programs with zero quota (there are some programs like this - BAD DATA):
df_programs_neat = df_programs_neat[df_programs_neat['quota'] > 0]

# SANITY CHECK: 'amount_matched' should be no more than 'wished_by'
#df_programs_neat[df_programs_neat['amount_matched'] > df_programs_neat['wished_by']] # is empty!

#------------------------ Adding 'how_selective' column to df_programs_neat: #(students wished this program)/quota:
df_programs_neat['how_selective'] = (
    df_programs_neat['wished_by'].div(df_programs_neat['quota']).replace([np.inf, -np.inf], np.nan)
)

# ================================= Data for resident trait for participated programs and ranked by them students =============================
# Renaming columns:
rename_map = {
    'G_CN_COD': 'student_id',
    'G_TI_COD': 'TI_id',
    'I_IS_FLC_SEC': 'resident'
}
df_resident = i_ins.rename(columns=rename_map)
cols_to_keep = ['student_id', 'TI_id', 'resident']
df_resident = df_resident[cols_to_keep]
df_resident = df_resident[df_resident['resident'] == 1] # leaving only rows with residents
df_resident = (
    df_resident[df_resident['TI_id'].isin(df_programs['TI_id'])].copy().reset_index(drop=True)
) # leaving only proposed programs' data
cols_to_keep = ['student_id', 'TI_id']
df_resident = df_resident[cols_to_keep] # now it contains a row only if student is a resident
df_resident = (
    df_resident.merge(df_programs[['TI_id', 'program_id']].drop_duplicates(), on='TI_id', how='left')
    [['TI_id', 'program_id', 'student_id']].reset_index(drop=True)
) # expanding by including program_id
cols_to_keep = ['program_id', 'student_id']
df_resident = df_resident[cols_to_keep] # now it contains a row only program and sudent ids

#leave only rows where a student was ranked by a program (that participated in main phase):
pairs = df_ranks[['program_id', 'student_id']].drop_duplicates()
df_resident = (
    df_resident.merge(pairs, on=['program_id', 'student_id'], how='inner').reset_index(drop=True)
)

# ================================= Data for wishes of students ranked by participated programs =============================
cols_to_keep = ['student_id', 'program_id']
df_wishes = df_ranks[cols_to_keep].drop_duplicates().reset_index(drop=True) # dont need a_voe data

# ================================= Data for all students ranked by at least one participated program =============================
# Renaming columns:
rename_map = {
    'G_CN_COD': 'student_id',
    'G_CN_BRS': 'b_trait',
    'G_CN_SEX_FEM': 'sex', # 1 - female
    'G_CN_NAT': 'is_french', # F - french
    'G_CN_COD_POS': 'post_code'
}
df_students = g_can.rename(columns=rename_map)
cols_to_keep = ['student_id', 'b_trait', 'sex', 'is_french', 'post_code']
df_students = df_students[cols_to_keep]
df_students = df_students[df_students['student_id'].isin(df_ranks['student_id'])] # keeping only students ranked by participated programs
df_students['is_french'] = (
    df_students['is_french'].replace({'F': 1, 'E': 0}).astype('int8')
)
df_students['b_trait'].replace(2, 0, inplace=True) # 2 plays no role

# ----------------------------- Adding column with total number of wishes:
wishes_per_student = df_ranks['student_id'].value_counts() # how many wishes each student has
df_students['wishes_count'] = (
    df_students['student_id'].map(wishes_per_student).fillna(0).astype(int)
)

# ----------------------------- Adding column with total number of b-only-wishes:
b_only_programs = (
    df_programs_neat.loc[df_programs_neat['b_only_type'].eq(1), 'program_id']
)
mask = df_ranks['program_id'].isin(b_only_programs)
b_counts = (
    df_ranks[mask].groupby('student_id').size()
)
df_students['b_only_wishes_count'] = (
    df_students['student_id'].map(b_counts).fillna(0).astype(int)
)

# ----------------------------- Adding column with total number of br-wishes:
br_programs = (
    df_programs_neat.loc[df_programs_neat['br_type'].eq(1), 'program_id']
)
mask = df_ranks['program_id'].isin(br_programs)
br_counts = (
    df_ranks[mask].groupby('student_id').size()
)
df_students['br_wishes_count'] = (
    df_students['student_id'].map(br_counts).fillna(0).astype(int)
)

# ----------------------------- Adding column with total number of br-wishes with residentships:
rank_filtered = (
    df_ranks.merge(br_programs.drop_duplicates().to_frame(), on='program_id', how = 'inner')
    .merge(df_resident, on=['student_id', 'program_id'], how = 'inner')
)
br_res_counts = (
    rank_filtered.groupby('student_id').size()
)
df_students['br_wishes_resident_count'] = (
    df_students['student_id'].map(br_res_counts).fillna(0).astype(int)
)

# ----------------------------- Adding column with total number of internat-wishes:
internat_programs = (
    df_programs_neat.loc[df_programs_neat['internat'].eq(1), 'program_id']
)
mask = df_ranks['program_id'].isin(internat_programs)
internat_counts = (
    df_ranks[mask].groupby('student_id').size()
)
df_students['internat_wishes_count'] = (
    df_students['student_id'].map(internat_counts).fillna(0).astype(int)
)

#endregion

#====================================================================================================
#====================================PREPARING DATA FOR FGH ESTIMATOR================================
#====================================================================================================

#region Formatting all data tables into FGH-compatibe csv
p = Path(DATADIR)
data_programs = df_programs_neat.reset_index(drop=True)
data_programs = data_programs.drop(columns=['amount_matched', 'no_aa_type', 'how_selective']).drop_duplicates() # drop: no_aa - for no colinerity, amount_matched - since students cannot know this in advance, how_selective - info is in wished_by
data_programs['kind'] = pd.to_numeric(
    data_programs['kind'], errors='coerce'
).astype('int64')
data_programs['b_share'] = data_programs['b_share'].astype('int64')
data_programs['r_share'] = data_programs['r_share'].astype('int64')
data_programs.to_csv(p/"data_programs.csv", index = False)

data_students = df_students.reset_index(drop=True)
data_students['post_code'] = (
    data_students['post_code'].astype(str).str.replace(r'^[+-]?inf$', '', regex=True).replace(r'\D', '', regex=True)
)
data_students['post_code'] = pd.to_numeric(
    data_students['post_code'], errors='coerce'
).astype('Int64')
data_students['post_code'] = data_students['post_code'].fillna(0) # all not converted post codes are 0
data_students['sex'] = data_students['sex'].astype('int64')
data_students['b_trait'] = data_students['b_trait'].fillna(0).astype('int64')
data_students.to_csv(p/"data_students.csv", index = False)

df_matching.to_csv(p/"data_matching.csv", index = False)

df_ranks['merit_rank'] = df_ranks['merit_rank'].astype('int64')
df_ranks['call_rank'] = df_ranks['call_rank'].astype('int64')
df_ranks.to_csv(p/"data_ranks.csv", index = False)

df_resident.to_csv(p/"data_resident.csv", index = False)

df_wishes.to_csv(p/"data_wishes.csv", index = False)

#endregion
