import pandas as pd
import numpy as np
import pickle
import gc
from datetime import datetime
import warnings

#region PCS 2022 DATA PREPROCESSING

#f = open(r"C:\Users\Public\Documents\DenisProj\a_adm.csv","r")

# Candidates data on b trait ---------------------------------------------------------
read_g_can = False # g_can is loaded from preprocessed pkl file

if read_g_can:
    g_can_raw = pd.read_csv('g_can.csv',encoding = "ISO-8859-1")
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

# G_CN_BRS - 0 - non boursier, 1 - secondary school b., 2 - higher education b.
# G_CN_FLG_RA - 1 if candidate activated automatic responder
number_brs_sec = (g_can['G_CN_BRS'] == 1).sum() # 199610 boursiers sec. shool
number_brs_high = (g_can['G_CN_BRS'] == 2).sum() # 152820 boursiers high. education

# Data on propositions ------------------------------------------------------
a_adm = pd.read_csv('a_adm.csv',encoding = "ISO-8859-1")
#print(a_adm.dtypes)

#How many proposed wishes were with accomodation (1.6%)
np.mean(a_adm['I_RH_COD'])

#Extracting times of all propositions and converting into data format
dates_of_propositions_STR = a_adm['A_AM_DAT']
#dates_of_propositions_no_time_STR = [s[:9] for s in dates_of_propositions_STR]
dates_of_propositions = pd.to_datetime(dates_of_propositions_STR, format='%d%b%Y:%H:%M:%S')
#print(dates_of_propositions)

#Extracting times of all responces and converting into data format
dates_of_responces_STR = a_adm['A_AM_DAT_REP']
dates_of_responces = pd.to_datetime(dates_of_responces_STR, format='%d%b%Y:%H:%M:%S')
#print(dates_of_responces)

#Stacking all propositions and responces into n-by-2 dataframe
dates_propositions_and_responces = np.column_stack((dates_of_propositions,dates_of_responces))
print(dates_propositions_and_responces)
df_propositions = pd.DataFrame(dates_propositions_and_responces, columns=["Proposition","Responce"])
#print(df_propositions)

#PROBLEM1: sometimes time of responce deadline ('A_AM_DAT_REP_LIM') is earlier then time of proposition ('A_AM_DAT') (e.g. for index 3877081)
#PROBLEM1 SOLVED: I checked below that responce time ('A_AM_DAT_REP') (if not NaN) is always weakly after proposition time:
df_propositions['Proposition_earlier'] = df_propositions['Proposition'] <= df_propositions['Responce']
filtered_df = df_propositions[(df_propositions['Responce'].notna()) & (df_propositions['Proposition_earlier'] == False)]
#filtered_df is empty, i.e. responce time is not weakly after than proposition time only if there was no responce
#print(filtered_df)

#How many proposals were not responded: 9312 out of 3877697 (0.24%)
df_propositions_no_responce = df_propositions[pd.isna(df_propositions['Responce'])]
share_of_not_responded_propositions = df_propositions_no_responce.index.size/dates_of_propositions.index.size

#---------------------------------------------------------------------------------------
#Data mining on internats:

#How many courses have a boarding (boys, girls, or mixed)
a_rec_raw = pd.read_csv('a_rec.csv',encoding = "ISO-8859-1")
#PROBLEM2: csv file contains errors: text got inserted into the first column ('G_TI_COD')
a_rec = a_rec_raw[a_rec_raw['G_TA_COD'].map(a_rec_raw['G_TA_COD'].value_counts()) == 1] #PROBLEM2 SOLVED: leave only rows with unique entry in the column 'G_TA_COD', now # of rows is the same as in the initial STATA file 'a_rec'
internats_all =  a_rec[(a_rec['A_RC_CAP_INT_GAR'].notna()) | (a_rec['A_RC_CAP_INT_FIL'].notna()) | (a_rec['A_RC_CAP_INT_MIX'].notna())] #619 internats (formations with some dorm quota)
share_of_internats = internats_all.index.size/a_rec.index.size #2.7% of all courses are internats
internats_mixed = a_rec[a_rec['A_RC_CAP_INT_MIX'].notna()] #168 internats
internats_boys_and_girls = a_rec[(a_rec['A_RC_CAP_INT_GAR'].notna()) & (a_rec['A_RC_CAP_INT_FIL'].notna())] #412 internats
internats_only_boys = a_rec[(a_rec['A_RC_CAP_INT_GAR'].notna()) & (pd.isna(a_rec['A_RC_CAP_INT_FIL']))] #19 internats
internats_only_girls = a_rec[(a_rec['A_RC_CAP_INT_FIL'].notna()) & (pd.isna(a_rec['A_RC_CAP_INT_GAR']))] #20 internats

# No internat has a quota for a gender and a mixed quota
internats_boys_or_girls_and_mixed = a_rec[((a_rec['A_RC_CAP_INT_GAR'].notna()) | (a_rec['A_RC_CAP_INT_FIL'].notna())) & (a_rec['A_RC_CAP_INT_MIX'].notna())] #0 internats

#How many propositions were made by internats (propositions with [a_adm['I_RH_COD'] == 1] should have been made only by internats)
df_propositions_by_internats = a_adm[a_adm['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 254111
# how many of them were made with a room?
num_of_propositions_by_internats_with_room = (df_propositions_by_internats['I_RH_COD'] == 1).sum() # 44606 made by internats
num_of_propositions_with_room = (a_adm['I_RH_COD'] == 1).sum() # 60323 made in total (PROBLEM3: 15k were made by whom?????)
# lets find institutions which made a proposition with a room and do not have a dorm quota in the data
propositions_by_internats_with_room = df_propositions_by_internats[df_propositions_by_internats['I_RH_COD'] == 1]
propositions_with_room = a_adm[a_adm['I_RH_COD'] == 1]
# are 'G_TI_COD' and 'G_TA_COD' the same:
num_of_G_TI_COD_a_adm = a_adm['G_TI_COD'].nunique() # 16821
num_of_G_TA_COD_a_adm = a_adm['G_TA_COD'].nunique() # 18764 institutions made at least one proposition (out of 22811)
num_of_G_TI_COD_a_rec = a_rec['G_TI_COD'].nunique() # 20840
num_of_G_TA_COD_a_rec = a_rec['G_TA_COD'].nunique() # 22811
# so, no, sometimes they are different; moreover, NOTICE: around 4k institutions did not make a single proposal in 2022
# should be CLAIM1: it is possible that there are multiple 'G_TA_COD' for one 'G_TI_COD', but not vise-versa!
# checking it for a_rec:
grouped_G_TI_COD_a_rec = a_rec.groupby('G_TI_COD')['G_TA_COD'].nunique()
num_of_blocks_G_TI_COD_a_rec = grouped_G_TI_COD_a_rec[grouped_G_TI_COD_a_rec > 1].count() # 815 blocks of the same TI and different TA
grouped_G_TA_COD_a_rec = a_rec.groupby('G_TA_COD')['G_TI_COD'].nunique()
num_of_blocks_G_TA_COD_a_rec = grouped_G_TA_COD_a_rec[grouped_G_TA_COD_a_rec > 1].count() # 0 blocks of the same TI and different TA
# checking it for a_adm:
grouped_G_TI_COD_a_adm = a_adm.groupby('G_TI_COD')['G_TA_COD'].nunique()
num_of_blocks_G_TI_COD_a_adm = grouped_G_TI_COD_a_adm[grouped_G_TI_COD_a_adm > 1].count() # 795 blocks of the same TI and different TA
grouped_G_TA_COD_a_adm = a_adm.groupby('G_TA_COD')['G_TI_COD'].nunique()
num_of_blocks_G_TA_COD_a_adm = grouped_G_TA_COD_a_adm[grouped_G_TA_COD_a_adm > 1].count() # 0 blocks of the same TI and different TA
# Thus, CLAIM1 is verified
# Back to PROBLEM3: find all 'G_TA_COD', s.t. it made a proposition with a room but is not in my list of internats
propositions_by_NOT_internats_with_room = propositions_with_room[~propositions_with_room['G_TA_COD'].isin(propositions_by_internats_with_room['G_TA_COD'])] # exactly 15717 left propositions
# TA of the first entry is 2112, here are it quotas from a_rec (should be no quotas)
institution_with_G_TA_COD_2112_a_rec_row = a_rec[a_rec['G_TA_COD'] == 2112]
#print(institution_with_G_TA_COD_2112_a_rec_row[['A_RC_CAP_INT_GAR','A_RC_CAP_INT_FIL','A_RC_CAP_INT_MIX']])
# PROBLEM3: So, for instance, for institution with G_TA_COD = 2112 we dont have info on its dorm quotas!
# So, lets create a list of internats without data for dorm quotas (which proposed with a room):
institutions_proposed_room = a_rec[a_rec['G_TA_COD'].isin(propositions_with_room['G_TA_COD'])] # all institutions that proposed room at least once
institutions_proposed_room_and_have_data_on_dorm_quota = institutions_proposed_room[institutions_proposed_room['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 614 out of 619 internats with data on dorm quotas made at leats one proposition with a room (all 619 proposed though)
institutions_proposed_room_and_dont_have_data_on_dorm_quota = institutions_proposed_room[~institutions_proposed_room['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 138 (potential) internats dont have data on dorm quotas: BAD_DATA
BAD_DATA_institutions = institutions_proposed_room_and_dont_have_data_on_dorm_quota

# excluding bad data formations from dfs:
a_adm_cleared = a_adm[~a_adm['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]
a_rec_cleared = a_rec[~a_rec['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]

#---------------------------------------------------------------------------------------

#NOTICE: in a_rec sometimes quota of an institution 'A_RC_CAP' is nan, while it is not nan at 'A_RC_CAP_INF' => should treat 'A_RC_CAP_INF' as total quota (?)
count = a_rec_cleared[(a_rec_cleared['A_RC_CAP'].notna()) & (a_rec_cleared['A_RC_CAP_INF'].isna())].shape[0] # if 'A_RC_CAP' is not nan, then 'A_RC_CAP_INF' is also not nan - it is good
institutions_with_two_quotas = a_rec_cleared[(a_rec_cleared['A_RC_CAP'].notna()) & (a_rec_cleared['A_RC_CAP'] != a_rec_cleared['A_RC_CAP_INF'])] # for 19 institutions 'A_RC_CAP_INF' != 'A_RC_CAP' => dont know which quota to use: probably, need to use 'A_RC_CAP_INF', or treat as BAD_DATA
institutions_with_no_quota = a_rec_cleared[(a_rec_cleared['A_RC_CAP_INF'].isna()) & (a_rec_cleared['A_RC_CAP'].isna())] # for 1047 institutions 'A_RC_CAP_INF' and 'A_RC_CAP' are nan => dont know the quota: BAD_DATA
# adding them to BAD_DATA_institutions:
BAD_DATA_institutions = pd.concat([BAD_DATA_institutions, institutions_with_no_quota], ignore_index=True)

# excluding bad data formations from dfs:
a_adm_cleared = a_adm[~a_adm['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]
a_rec_cleared = a_rec[~a_rec['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]

# going back to internats: quota of the mixed internat should not be greater than amount of mixed rooms:
internats_mixed_BAD = internats_mixed[internats_mixed['A_RC_CAP_INT_MIX'] > internats_mixed['A_RC_CAP']] #40 out of 168 mixed internats have more rooms than its qouta: BAD_DATA (probably there are common dorms that we dont know how to deal with)
# quota of the boys internat should not be greater than amount of boys rooms:
internats_only_boys_BAD = internats_only_boys[internats_only_boys['A_RC_CAP_INT_GAR'] > internats_only_boys['A_RC_CAP']] #0 boys internats have more rooms than its qouta
# quota of the girls internat should not be greater than amount of girls rooms:
internats_only_girls_BAD = internats_only_girls[internats_only_girls['A_RC_CAP_INT_FIL'] > internats_only_girls['A_RC_CAP']] #0 girls internats have more rooms than its qouta
# quota of the boys and girls internat should not be greater than amount of girls rooms and amount of boys rooms:
internats_boys_and_girls_BAD = internats_boys_and_girls[(internats_boys_and_girls['A_RC_CAP_INT_FIL'] > internats_boys_and_girls['A_RC_CAP']) & (internats_boys_and_girls['A_RC_CAP_INT_GAR'] > internats_boys_and_girls['A_RC_CAP'])] #51 out of 412 of boys and girls internats have more rooms than both qoutas: BAD_DATA (probably there are common dorms that we dont know how to deal with)
internats_boys_and_girls_MAYBE_BAD = internats_boys_and_girls[(internats_boys_and_girls['A_RC_CAP_INT_FIL'] > internats_boys_and_girls['A_RC_CAP']) | (internats_boys_and_girls['A_RC_CAP_INT_GAR'] > internats_boys_and_girls['A_RC_CAP'])] #88 out of 412 of boys and girls internats have more rooms than at least one of its qoutas: maybe BAD_DATA (probably there are also common dorms that we dont know how to deal with)
internats_boys_and_girls_MAYBE_BAD_ALSO = internats_boys_and_girls[(internats_boys_and_girls['A_RC_CAP_INT_FIL'] + internats_boys_and_girls['A_RC_CAP_INT_GAR']) > internats_boys_and_girls['A_RC_CAP']] #117 out of 412 of boys and girls internats have more rooms than sum of its qoutas: maybe also BAD_DATA (probably there are also common dorms that we dont know how to deal with)

# adding to BAD_DATA_institutions:
BAD_DATA_institutions = pd.concat([BAD_DATA_institutions, internats_mixed_BAD, internats_boys_and_girls_BAD], ignore_index=True)

# excluding bad data formations from dfs:
a_adm_cleared = a_adm[~a_adm['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]
a_rec_cleared = a_rec[~a_rec['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])]

#How many courses use Affirmative Action (AA) (b and/or r traits)
# In a_rec table:
# 'A_RC_FLG_TAU_BRS' - flag whether it cares about b (1) or not (0);
# 'A_RC_TAU_BRS_REC' - min percentage of boursiers
# 'A_RC_TAU_BRS_OTO' - min percentage of boursiers (automatically calculated ????)
# 'A_RC_FLG_TAU_NON_RES' - flag whether it cares about r (1) or not (0);
# 'A_RC_TAU_NON_RES_REC' - (100 - min percentage of residents)
# 'A_RC_TAU_NON_RES_OTO' - (100 - min percentage of residents) (automatically calculated ????)
# 3rd trait???????:
# 'A_RC_FLG_TAU_PRO' - flag whether it cares about 'candidats en bacs pro' (candidates for vocational bac) (1) or not (0);
# 'A_RC_TAU_PRO_REC' - (100 - min percentage of pro)
# 'A_RC_TAU_PRO_OTO' - (100 - min percentage of pro) (automatically calculated ????)

# First, I check whether flag = 1 implies that corresponding percentage is in the data:
# for b:
institutions_care_for_b_no_base_bquota = a_rec[(a_rec['A_RC_FLG_TAU_BRS']==1) & (a_rec['A_RC_TAU_BRS_REC'].isna())] # 224 institutions that care about b dont have 'A_RC_TAU_BRS_REC', but always have 'A_RC_TAU_BRS_OTO':
institutions_care_for_b_no_bquota = a_rec[(a_rec['A_RC_FLG_TAU_BRS']==1) & (a_rec['A_RC_TAU_BRS_REC'].isna()) & (a_rec['A_RC_TAU_BRS_OTO'].isna())]
institutions_dont_care_for_b_has_bquota = a_rec[((a_rec['A_RC_FLG_TAU_BRS']==0) | (a_rec['A_RC_FLG_TAU_BRS'].isna())) & (a_rec['A_RC_TAU_BRS_REC'].notna())] # no institutions who dont care about b has target b quota; 5 has automatically calculated target quota though:
institutions_dont_care_for_b_has_aut_bquota = a_rec[((a_rec['A_RC_FLG_TAU_BRS']==0) | (a_rec['A_RC_FLG_TAU_BRS'].isna())) & (a_rec['A_RC_TAU_BRS_OTO'].notna())] # BAD_DATA

# adding to BAD_DATA_institutions:
BAD_DATA_institutions = pd.concat([BAD_DATA_institutions, institutions_dont_care_for_b_has_aut_bquota], ignore_index=True)

# for r:
institutions_care_for_r_no_base_rquota = a_rec[(a_rec['A_RC_FLG_TAU_NON_RES']==1) & (a_rec['A_RC_TAU_NON_RES_REC'].isna())] # 68 institutions that care about r dont have 'A_RC_TAU_NON_RES_REC', but always have 'A_RC_TAU_NON_RES_OTO':
institutions_care_for_r_no_rquota = a_rec[(a_rec['A_RC_FLG_TAU_NON_RES']==1) & (a_rec['A_RC_TAU_NON_RES_REC'].isna()) & (a_rec['A_RC_TAU_NON_RES_OTO'].isna())]
institutions_dont_care_for_r_has_rquota = a_rec[((a_rec['A_RC_FLG_TAU_NON_RES']==0) | (a_rec['A_RC_FLG_TAU_NON_RES'].isna())) & ((a_rec['A_RC_TAU_NON_RES_REC'].notna()) | ((a_rec['A_RC_TAU_NON_RES_OTO'].notna())))] # no institutions who dont care about r has target quota (0 has automatically calculated target quota)

# for pro......

#Second, I check whether target quotas for b = 0 for some institutions that care about b (shouldn't happen):
institutions_zero_bquota = a_rec[(a_rec['A_RC_FLG_TAU_BRS']==1) & ((a_rec['A_RC_TAU_BRS_REC']==0) | (a_rec['A_RC_TAU_BRS_OTO']==0))] # 288 have one of quotas for b as zero
# and whether target quotas for r = 0 for some institutions that care about r (shouldn't happen):
institutions_zero_rquota = a_rec[(a_rec['A_RC_FLG_TAU_NON_RES']==1) & ((a_rec['A_RC_TAU_NON_RES_REC']==0) | (a_rec['A_RC_TAU_NON_RES_OTO']==0))] # 75 have one of quotas for b as zero
# how many institutions have different quotas for b or r (base and automatically calculated)
institutions_different_brquotas = a_rec[((a_rec['A_RC_FLG_TAU_BRS']==1) & (a_rec['A_RC_TAU_BRS_REC'] != a_rec['A_RC_TAU_BRS_OTO'])) | ((a_rec['A_RC_FLG_TAU_NON_RES']==1) & (a_rec['A_RC_TAU_NON_RES_REC'] != a_rec['A_RC_TAU_NON_RES_OTO']))] # be careful with them for call order calculation: need to determine which quota is the correct one

# Third, how many institutions care for b:
institutions_care_only_for_b = a_rec[(a_rec['A_RC_FLG_TAU_BRS']==1) & ((a_rec['A_RC_FLG_TAU_NON_RES']==0) | (a_rec['A_RC_FLG_TAU_NON_RES'].isna()))] # 9739 out of 22811
institutions_proposed_care_only_for_b = institutions_care_only_for_b[institutions_care_only_for_b['G_TA_COD'].isin(a_adm['G_TA_COD'])] # 9680 of them proposed
internats_proposed_care_only_for_b = institutions_proposed_care_only_for_b[institutions_proposed_care_only_for_b['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 619 of the proposed ones are internats (all internats only care about b)
# for r:
institutions_care_only_for_r = a_rec[(a_rec['A_RC_FLG_TAU_NON_RES']==1) & ((a_rec['A_RC_FLG_TAU_BRS']==0) | (a_rec['A_RC_FLG_TAU_BRS'].isna()))] # 0 out of 22811
# for b and r:
institutions_care_for_br = a_rec[(a_rec['A_RC_FLG_TAU_NON_RES']==1) & (a_rec['A_RC_FLG_TAU_BRS']==1)] # 2839 out of 22811
institutions_proposed_care_for_br = institutions_care_for_br[institutions_care_for_br['G_TA_COD'].isin(a_adm['G_TA_COD'])] # 2839 of them proposed
internats_proposed_care_for_br = institutions_proposed_care_for_br[institutions_proposed_care_for_br['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 0 of the proposed ones are internats

# Lets see distribution of b and r quotas:
bins_bquota = a_rec['A_RC_TAU_BRS_REC'].value_counts().sort_index()
bins_rquota = ((a_rec['A_RC_TAU_NON_RES_REC'] * -1) + 100).value_counts().sort_index() # initial quotas are for max number of residents
bins_aut_bquota = a_rec['A_RC_TAU_BRS_OTO'].value_counts().sort_index()
bins_aut_rquota = a_rec['A_RC_TAU_NON_RES_OTO'].value_counts().sort_index()

# Lets see distribution of b quota for b-only-caring institutions:
bins_bquota_care_only_for_b = institutions_care_only_for_b['A_RC_TAU_BRS_REC'].value_counts().sort_index()

# Lets see distribution of b quota for br-caring institutions:
bins_bquota_care_for_br = institutions_care_for_br['A_RC_TAU_BRS_REC'].value_counts().sort_index()

# Distributions of seat quotas:
number_of_seats_total = a_rec_cleared['A_RC_CAP_INF'].sum(skipna = True)
number_of_candidates_proposed_to = a_adm_cleared['G_CN_COD'].nunique(dropna = True)
number_of_candidates_total = g_can.shape[0]
bins_seat_quota = a_rec_cleared['A_RC_CAP'].value_counts().sort_index() # for all formations
bins_seat_quota_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'].isin(institutions_proposed_care_only_for_b['G_TA_COD']),'A_RC_CAP'].value_counts().sort_index() # for b-caring formations
bins_seat_quota_br = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'].isin(institutions_proposed_care_for_br['G_TA_COD']),'A_RC_CAP'].value_counts().sort_index() # for br-caring formations
number_of_seats_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'].isin(institutions_proposed_care_only_for_b['G_TA_COD']),'A_RC_CAP_INF'].sum(skipna = True)
number_of_seats_br = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'].isin(institutions_proposed_care_for_br['G_TA_COD']),'A_RC_CAP_INF'].sum(skipna = True)

#Groups data (are they the real colleges here?)-------------------------------------------------------
a_rec_grp = pd.read_csv('a_rec_grp.csv',encoding = "ISO-8859-1") # contains group-institution corresp. and quotas of groups
number_of_seats_grp_total = a_rec_grp['A_RG_PLA'].sum(skipna = True)

# Can it be: one group contains more then one G_TI_COD?:
a_rec_grp_same_g_different_inst = (a_rec_grp.groupby('C_GP_COD')['G_TI_COD'].nunique() > 1).any() # NO! One group contains at most one upper-institution

# Each row of 'a_rec_grp' per one group?
number_of_institutions_group_data = a_rec_grp['G_TA_COD'].nunique() # each initial institution has at least one group
number_of_groups = a_rec_grp['C_GP_COD'].nunique() # more groups than rows, maybe need to remove all groups with no seat quota:
a_rec_grp_no_nan_quota = a_rec_grp.dropna(subset = ['A_RG_PLA']) # dropped all rows without seat quota
a_rec_grp_no_nan_quota = a_rec_grp_no_nan_quota[a_rec_grp_no_nan_quota['A_RG_PLA'] != 0] # dropped all rows with 0 seat quota
number_of_groups_no_nan_quota = a_rec_grp_no_nan_quota['C_GP_COD'].nunique() # still more rows of data than groups
a_rec_grp_no_nan_quota_no_BAD_DATA_inst = a_rec_grp_no_nan_quota[~a_rec_grp_no_nan_quota['G_TA_COD'].isin(BAD_DATA_institutions['G_TA_COD'])] # removing all rows with BAD INSTITUTIONS
number_of_groups_no_nan_quota_no_BAD_DATA_inst = a_rec_grp_no_nan_quota_no_BAD_DATA_inst['C_GP_COD'].nunique() # by (30 529 - 29 295) less then number of rows
a_rec_grp_no_nan_quota_no_BAD_DATA_inst_unique_groups = a_rec_grp_no_nan_quota_no_BAD_DATA_inst[~a_rec_grp_no_nan_quota_no_BAD_DATA_inst['C_GP_COD'].duplicated(keep=False)] # removing all rows where the same group meets more than once
# NOW: a_rec_grp_no_nan_quota_no_BAD_DATA_inst_unique_groups contains one unique group per row (no two rows contain the same group C_GP_COD)

# How many formations have their own group?
unique_once_values = a_rec_grp['G_TA_COD'].value_counts()[a_rec_grp['G_TA_COD'].value_counts() == 1].index
institutions_with_one_group = a_rec_grp[a_rec_grp['G_TA_COD'].isin(unique_once_values)]
number_of_institutions_with_one_group = (a_rec_grp['G_TA_COD'].value_counts() == 1).sum()

# Rankings for groups ---------------------------------------------------
c_can_grp = pd.read_csv('c_can_grp.csv',encoding = "ISO-8859-1") # contains ranking and call order for each group
candidates_in_groups_rankings = c_can_grp['G_CN_COD'].unique() # 938 639 all candidates from rankings of groups
candidates_in_groups_rankings_with_b = g_can.loc[g_can['G_CN_COD'].isin(candidates_in_groups_rankings), 'G_CN_BRS'].sum()
c_can_grp_no_nan_rank = c_can_grp.dropna(subset = ['C_CG_RAN']) # dropped all rows without initial ranks of candidates
c_can_grp_processed = c_can_grp_no_nan_rank[c_can_grp_no_nan_rank['C_GP_COD'].isin(a_rec_grp_no_nan_quota_no_BAD_DATA_inst_unique_groups['C_GP_COD'])] # leaving only data for groups with one institution

# Resident trait data -------------------------------------------------------------------
read_i_ins = False # i_ins is loaded from preprocessed pkl file

if read_g_can:
    i_ins = pd.read_csv('i_ins.csv',encoding = "ISO-8859-1") # should be 10 415 378 rows

    # SAVING i_ins:
    with open("i_ins.pkl", "wb") as f:
        pickle.dump(i_ins, f)

# READING i_ins.pkl:
with open("i_ins.pkl", "rb") as f:
    i_ins = pickle.load(f)

# I_IS_FLC_SEC - binary, should indicate whether a candidate G_CN_COD is a resident for an institution G_TI_COD
# I_IS_VAL - binary, indicates whether application is validated for a candidate G_CN_COD and an institution G_TI_COD (?)

# For each candidate: how many residentships does he have?
residentships_data = i_ins.groupby('G_CN_COD')['I_IS_FLC_SEC'].sum()
# Distribution for number of residentships:
bins_residentship_per_candidate = residentships_data.iloc[:].value_counts().sort_index()

# DELETE VAR:
#del VAR
#gc.collect()

# Wishes data ----------------------------------------------------------------------------------------
a_voe = pd.read_csv('a_voe.csv',encoding = "ISO-8859-1") # should be 19 563 592 rows

# How many wishes does each candidate have?
wishes_data = a_voe['G_CN_COD'].value_counts()
wishes_data.name = "number_of_wishes"
# Distribution for number of wishes:
bins_wishes_per_candidate = wishes_data.iloc[:].value_counts().sort_index()

#endregion

#region CALL ORDER CALCULATION AND COMPARISON for each group C_GP_COD from a_rec_grp_no_nan_quota_no_BAD_DATA_inst_unique_groups
a_rec_grp_curr = a_rec_grp_no_nan_quota_no_BAD_DATA_inst_unique_groups # shorten the name

def call_order(candidates_f, preferences_f, q, share_b, share_r):
    share_b = np.nan_to_num(share_b) # makes 0 from nan
    share_r = np.nan_to_num(share_r) # makes 0 from nan
    curr_call_order_rank = 1

    # Step 1:
    if ((share_b == 0) & (share_r == 0)):
        preferences_f['Call_order_Mine'] = preferences_f['C_CG_RAN']
        return preferences_f['Call_order_Mine']
    elif ((share_b > 0) & (share_r == 0)):
        candidates_b = preferences_f[preferences_f['Boursier'] == 1] # seems like only candidates with 1 (not 2) are boosted in call order
        if candidates_b.empty:
            preferences_f.loc[0,'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1
        else:
            best_candidate_b = candidates_b.loc[candidates_b.index.min()]
            best_candidate_b = best_candidate_b['G_CN_COD']
            preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_b, 'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1
    elif ((share_b == 0) & (share_r > 0)):
        candidates_r = preferences_f[preferences_f['Resident'] == 1]
        if candidates_r.empty:
            preferences_f.loc[0,'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1
        else:
            best_candidate_r = candidates_r.loc[candidates_r.index.min()]
            best_candidate_r = best_candidate_r['G_CN_COD']
            preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_r, 'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1
    elif ((share_b > 0) & (share_r > 0)):
        candidates_br = preferences_f[(preferences_f['Boursier'] == 1) & (preferences_f['Resident'] == 1)]
        if candidates_br.empty:
            candidates_b = preferences_f[preferences_f['Boursier'] == 1]
            if candidates_b.empty:
                candidates_r = preferences_f[preferences_f['Resident'] == 1]
                if candidates_r.empty:
                    preferences_f.loc[0,'Call_order_Mine'] = curr_call_order_rank
                    curr_call_order_rank += 1
                else:
                    best_candidate_r = candidates_r.loc[candidates_r.index.min()]
                    best_candidate_r = best_candidate_r['G_CN_COD']
                    preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_r, 'Call_order_Mine'] = curr_call_order_rank
                    curr_call_order_rank += 1
            else:
                best_candidate_b = candidates_b.loc[candidates_b.index.min()]
                best_candidate_b = best_candidate_b['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_b, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
        else:
            best_candidate_br = candidates_br.loc[candidates_br.index.min()]
            best_candidate_br = best_candidate_br['G_CN_COD']
            preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_br, 'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1

    # Step k:
    number_of_candidates_f = candidates_f.shape[0]
    for k in range(2,number_of_candidates_f + 1):
        Lb = preferences_f[(preferences_f['Boursier'] == 1) & (preferences_f['Call_order_Mine']).notna()]
        Lb = Lb.shape[0]
        Lr = preferences_f[(preferences_f['Resident'] == 1) & (preferences_f['Call_order_Mine']).notna()]
        Lr = Lr.shape[0]
        candidates_left = preferences_f[(preferences_f['Call_order_Mine']).isna()]

        if ((Lb/k >= share_b/100) & (Lr/k >= share_r/100)):
            best_candidate = candidates_left.loc[candidates_left.index.min()]
            best_candidate = best_candidate['G_CN_COD']
            preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate, 'Call_order_Mine'] = curr_call_order_rank
            curr_call_order_rank += 1
        elif ((Lb/k < share_b/100) & (Lr/k >= share_r/100)):
            candidates_b = candidates_left[candidates_left['Boursier'] == 1]
            if candidates_b.empty:
                best_candidate = candidates_left.loc[candidates_left.index.min()]
                best_candidate = best_candidate['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
            else:
                best_candidate_b = candidates_b.loc[candidates_b.index.min()]
                best_candidate_b = best_candidate_b['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_b, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
        elif ((Lb/k >= share_b/100) & (Lr/k < share_r/100)):
            candidates_r = candidates_left[candidates_left['Resident'] == 1]
            if candidates_r.empty:
                best_candidate = candidates_left.loc[candidates_left.index.min()]
                best_candidate = best_candidate['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
            else:
                best_candidate_r = candidates_r.loc[candidates_r.index.min()]
                best_candidate_r = best_candidate_r['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_r, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
        elif ((Lb/k < share_b/100) & (Lr/k < share_r/100)):
            candidates_br = candidates_left[(candidates_left['Boursier'] == 1) & (candidates_left['Resident'] == 1)]
            if candidates_br.empty:
                candidates_b = candidates_left[candidates_left['Boursier'] == 1]
                if candidates_b.empty:
                    candidates_r = candidates_left[candidates_left['Resident'] == 1]
                    if candidates_r.empty:
                        best_candidate = candidates_left.loc[candidates_left.index.min()]
                        best_candidate = best_candidate['G_CN_COD']
                        preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate, 'Call_order_Mine'] = curr_call_order_rank
                        curr_call_order_rank += 1
                    else:
                        best_candidate_r = candidates_r.loc[candidates_r.index.min()]
                        best_candidate_r = best_candidate_r['G_CN_COD']
                        preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_r, 'Call_order_Mine'] = curr_call_order_rank
                        curr_call_order_rank += 1
                else:
                    best_candidate_b = candidates_b.loc[candidates_b.index.min()]
                    best_candidate_b = best_candidate_b['G_CN_COD']
                    preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_b, 'Call_order_Mine'] = curr_call_order_rank
                    curr_call_order_rank += 1
            else:
                best_candidate_br = candidates_br.loc[candidates_br.index.min()]
                best_candidate_br = best_candidate_br['G_CN_COD']
                preferences_f.loc[preferences_f['G_CN_COD'] == best_candidate_br, 'Call_order_Mine'] = curr_call_order_rank
                curr_call_order_rank += 1
    return preferences_f['Call_order_Mine']

call_order_check = False

if call_order_check:

    counter_group = 0 # in total: 28 770 groups
    number_of_groups_with_different_call_orders = 0 # RESULT: NO groups have different call orders => may use only 'C_CG_ORD_APP'
    number_of_groups_with_same_ranking_and_call_order = 0 # RESULT: 11 309
    number_of_groups_AA_with_same_ranking_and_call_order = 0 # for groups with at least one target qouta (>0), RESULT: 2 674
    number_of_groups_AA = 0 # RESULT: 20 135
    number_of_groups_AA_aut = 0 # RESULT: 20 133
    number_of_groups_AA_not_aut = 0 # RESULT: 20 085
    number_of_groups_AA_aut_b = 0 # RESULT: 20 079

    groups_only_b_same = []
    groups_only_b_diff = []
    groups_br_same = []
    groups_br_diff = []

    for f in a_rec_grp_curr['C_GP_COD']: # for each group
        counter_group +=1
        print(f"=================== Group {f} ({counter_group} / {a_rec_grp_curr['C_GP_COD'].nunique()}) ==========================")
        G_TI_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TI_COD'].item()
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()
        q = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item()
        share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
        share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()
        share_b_aut = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_OTO'].item()
        share_r_aut = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_OTO'].item()

        # Extracting initial ranking and call order for f:
        group_curr_data = c_can_grp_processed[(c_can_grp_processed['C_GP_COD'] == f)].copy() # 'C_CG_RAN' - initial ranking, 'C_CG_ORD_APP' - call order, 'C_CG_ORD_APP_AFF' - final call order (on web-site)

        candidates_f = group_curr_data['G_CN_COD'] # all candidates from ranking of f
        candidates_f = candidates_f.reset_index(drop=True)

        print(f"# Seats = {int(q)}, # Candidates = {int(candidates_f.shape[0])}")
        print(f"           Quotas: for b = {share_b}%, for r = {share_r}%")
        print(f" Automatic quotas: for b = {share_b_aut}%, for r = {share_r_aut}%")

        #if not group_curr_data['C_CG_ORD_APP'].equals(group_curr_data['C_CG_ORD_APP_AFF']):
        #    number_of_groups_with_different_call_orders += 1 # RESULT: NO groups have different call orders => may use only 'C_CG_ORD_APP'

        preferences_f = group_curr_data[['G_CN_COD', 'C_CG_RAN', 'C_CG_ORD_APP']].copy()

        # For each relevant candidate: add b and r indicators:
        preferences_f[['Boursier', 'Resident', 'Res_VAL']] = None
        for c in candidates_f:
            preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Boursier'] = g_can.loc[g_can['G_CN_COD'] == c, 'G_CN_BRS'].item() # seems like 2 has no effect on call order (no boost)
            preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Resident'] = i_ins.loc[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'] == G_TI_COD_curr), 'I_IS_FLC_SEC'].item()
            preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Res_VAL'] = i_ins.loc[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'] == G_TI_COD_curr), 'I_IS_VAL'].item()

        preferences_f = preferences_f.sort_values(by='C_CG_RAN')
        preferences_f = preferences_f.reset_index(drop=True)
        preferences_f['Call_order_Mine'] = None

        if ((np.nan_to_num(share_b) > 0) | (np.nan_to_num(share_r) > 0)): # if there is AA
            if int(candidates_f.shape[0]) > 0: # if there is at least one candidate
                Call_order_Mine = call_order(candidates_f, preferences_f, q, share_b, share_r)

                # Checking whether my call order and one from data are the same:
                number_of_different_entries = (preferences_f['C_CG_ORD_APP'] != preferences_f['Call_order_Mine']).sum()
                if (number_of_different_entries == 0):
                    print("Call orders are the same!")
                    if ((np.nan_to_num(share_b) > 0) & (np.nan_to_num(share_r) == 0)):
                        groups_only_b_same.append(f)
                    if ((np.nan_to_num(share_b) > 0) & (np.nan_to_num(share_r) > 0)):
                        groups_br_same.append(f)
                else:
                    print(f"My call order differs in {number_of_different_entries} / {int(candidates_f.shape[0])} entries")
                    if ((np.nan_to_num(share_b) > 0) & (np.nan_to_num(share_r) == 0)):
                        groups_only_b_diff.append(f)
                    if ((np.nan_to_num(share_b) > 0) & (np.nan_to_num(share_r) > 0)):
                        groups_br_diff.append(f)
            else:
                print(f"There are no acceptable candidates!")
        else:
            print(f"There is no affirmative action!")

        print("-----------------------------------------------------------------------------------------")
        print(f"Numbers of AA groups (b-only, br):")
        print(f" Same call orders({len(groups_only_b_same)}, {len(groups_br_same)})")
        print(f" Diff call orders({len(groups_only_b_diff)}, {len(groups_br_diff)})")

        #if group_curr_data['C_CG_ORD_APP'].equals(group_curr_data['C_CG_RAN']):
        #    number_of_groups_with_same_ranking_and_call_order += 1

        #if (group_curr_data['C_CG_ORD_APP'].equals(group_curr_data['C_CG_RAN'])) & (((not np.isnan(share_b)) & (share_b>0)) | ((not np.isnan(share_b_aut)) & (share_b_aut>0)) | ((not np.isnan(share_r)) & (share_r>0)) | ((not np.isnan(share_r_aut)) & (share_r_aut>0))):
        #    number_of_groups_AA_with_same_ranking_and_call_order += 1

        #if ((not np.isnan(share_b)) & (share_b>0)) | ((not np.isnan(share_b_aut)) & (share_b_aut>0)) | ((not np.isnan(share_r)) & (share_r>0)) | ((not np.isnan(share_r_aut)) & (share_r_aut>0)): # from above: if f cares about trait x => it always has share_x_aut
        #    number_of_groups_AA += 1

        #if ((not np.isnan(share_b_aut)) & (share_b_aut>0)) | ((not np.isnan(share_r_aut)) & (share_r_aut>0)): # from above: if f cares about trait x => it always has share_x_aut
        #    number_of_groups_AA_aut += 1

        #if ((not np.isnan(share_b)) & (share_b>0)) | ((not np.isnan(share_r)) & (share_r>0)): # from above: if f cares about trait x => it always has share_x_aut
        #    number_of_groups_AA_not_aut += 1

        #if (not np.isnan(share_b_aut)) & (share_b_aut>0): # from above: if f cares about trait x => it always has share_x_aut, and if f cares about r => also cares about b
        #    number_of_groups_AA_aut_b += 1

        # Should be: number_of_groups_AA = number_of_groups_AA_aut = number_of_groups_AA_not_aut = number_of_groups_AA_aut_b - NOT TRUE (SEE ABOVE)

        # CONTINUE: calculate call order for approx 20k groups

    # SAVING lists of good and bad groups:
    #with open("groups_AA_call_order.pkl", "wb") as f:
    #     pickle.dump((groups_only_b_same, groups_only_b_diff, groups_br_same, groups_br_diff), f)

# LOADING lists of good and bad groups:
with open("groups_AA_call_order.pkl", "rb") as f:
    groups_only_b_same, groups_only_b_diff, groups_br_same, groups_br_diff = pickle.load(f)

#endregion

print(f"For b-only groups: {len(groups_only_b_same)} / {len(groups_only_b_diff) + len(groups_only_b_same)} are good ({100*len(groups_only_b_same)/(len(groups_only_b_diff) + len(groups_only_b_same)):.1f}%)")
print(f"For br groups: {len(groups_br_same)} / {len(groups_br_diff) + len(groups_br_same)} are good ({100*len(groups_br_same)/(len(groups_br_diff) + len(groups_br_same)):.1f}%)")
print(f"For all AA groups: {len(groups_br_same) + len(groups_only_b_same)} / {len(groups_br_diff) + len(groups_br_same) + len(groups_only_b_diff) + len(groups_only_b_same)} are good ({100*(len(groups_br_same) + len(groups_only_b_same))/(len(groups_br_diff) + len(groups_br_same) + len(groups_only_b_diff) + len(groups_only_b_same)):.1f}%)")

# IDEA: check for bad groups: are there pairs of candidates with the same sets of traits, s.t. their relative positions in call order and initial ranking are reversed?

#region MAIN DATA MINING: for all good groups we calculate # of (stage) blocking candidates for each time moment

# In table a_adm column 'A_SV_COD' stands for responces ids for a proposition, explanation of these ids are in table a_sit_voe

recalculate_prop_data = False

if recalculate_prop_data:
    counter_group = 0 # in total: 16 072 groups
    number_of_internats_b_only = 0 # 339 internats
    propositions_data_b_only = pd.DataFrame(columns = ['C_GP_COD','G_CN_COD','A_SV_COD','A_AM_DAT','A_AM_DAT_REP','Rank', 'Rank_CO','A_AM_DAT_NOTIME', 'A_AM_DAT_REP_NOTIME','Boursier', 'Resident','A_AM_DAT_NOTIME_DAYS_FROM_JAN1', 'A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1'])
    preferences_data_b_only = pd.DataFrame(columns = ['C_GP_COD','G_CN_COD','C_CG_RAN', 'C_CG_ORD_APP','Boursier', 'Resident'])
    for f in groups_only_b_same: # for each good b-only group
        counter_group +=1
        print(f"=================== Good b-only group {f} ({counter_group} / {len(groups_only_b_same)}) ==========================")
        G_TI_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TI_COD'].item()
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()

        if G_TA_COD_curr in internats_all['G_TA_COD'].values: # if this group is for internat
            number_of_internats_b_only +=1
            print("It is an internat!")
        else:
            q = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item()
            share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
            share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()

            # Extracting initial ranking and call order for f:
            group_curr_rank_data = c_can_grp_processed[(c_can_grp_processed['C_GP_COD'] == f)].copy() # 'C_CG_RAN' - initial ranking, 'C_CG_ORD_APP' - call order, 'C_CG_ORD_APP_AFF' - final call order (on web-site)

            candidates_f = group_curr_rank_data['G_CN_COD'] # all candidates from ranking of f
            candidates_f = candidates_f.reset_index(drop=True)

            print(f"# Seats = {int(q)}, # Candidates = {int(candidates_f.shape[0])}")
            print(f" Quotas: for b = {share_b}%, for r = {share_r}%")

            preferences_f = group_curr_rank_data[['C_GP_COD','G_CN_COD', 'C_CG_RAN', 'C_CG_ORD_APP']].copy()

            # For each relevant candidate: add b and r indicators:
            preferences_f[['Boursier', 'Resident']] = None
            for c in candidates_f:
                preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Boursier'] = g_can.loc[g_can['G_CN_COD'] == c, 'G_CN_BRS'].item() # seems like 2 has no effect on call order (no boost)
                preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Resident'] = i_ins.loc[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'] == G_TI_COD_curr), 'I_IS_FLC_SEC'].item()

            preferences_f['Boursier'] = preferences_f['Boursier'].replace(2,0) # deleting 2s from b trait
            preferences_f['Boursier'] = preferences_f['Boursier'].fillna(0) # changing all nans to 0 for b
            preferences_f['Resident'] = preferences_f['Resident'].fillna(0) # changing all nans to 0 for r
            preferences_f = preferences_f.sort_values(by='C_CG_RAN')
            preferences_f = preferences_f.reset_index(drop=True)

            # Extracting all propositions that the current group f made throughout the first stage (a_adm_cleared['A_TA_COD'] == 1) of Parcoursup
            group_curr_propositions_data_raw = a_adm_cleared[(a_adm_cleared['A_TA_COD'] == 1) & (a_adm_cleared['C_GP_COD'] == f)]

            # NOTE: no proposition in the first phase has nan in the responce date, ALSO no proposition in the first phase has A_SV_COD = -91, or -90, or -92, or 8 (cannot be proposed to type of agents)
            # 89 propositions have A_SV_COD = -39 (Fin procedure)

            # -------- Creating modified df with propositions data:
            group_curr_propositions_data = group_curr_propositions_data_raw[['C_GP_COD','G_CN_COD','A_SV_COD','A_AM_DAT','A_AM_DAT_REP']].copy()
            group_curr_propositions_data[['Rank', 'Rank_CO']] = None
            # copying ranking
            mapping = preferences_f.set_index('G_CN_COD')['C_CG_RAN']
            group_curr_propositions_data['Rank'] = group_curr_propositions_data['G_CN_COD'].map(mapping)
            # copying call order
            mapping = preferences_f.set_index('G_CN_COD')['C_CG_ORD_APP']
            group_curr_propositions_data['Rank_CO'] = group_curr_propositions_data['G_CN_COD'].map(mapping)

            # changing Status (A_SV_COD) for one of two states: accepted (1) or rejected (-1):
            group_curr_propositions_data.loc[group_curr_propositions_data['A_SV_COD'].isin([5,6,10]), 'A_SV_COD'] = 1
            group_curr_propositions_data.loc[group_curr_propositions_data['A_SV_COD'].isin([-31,-93,-50,-51,-53,-40,-41,-42,-30,-34,-25,-20,-21,-23,-24,-22,-43,-35,-36,-26,-18,-19,-27,-29,-28,-33,-94,-32,-39]), 'A_SV_COD'] = -1
            # what to do with A_SV_COD = -39 (Fin procedure)? lets put it into rejected

            group_curr_propositions_data['A_AM_DAT'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT'], format='%d%b%Y:%H:%M:%S')
            group_curr_propositions_data['A_AM_DAT_REP'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_REP'], format='%d%b%Y:%H:%M:%S')

            group_curr_propositions_data[['A_AM_DAT_NOTIME', 'A_AM_DAT_REP_NOTIME']] = None
            group_curr_propositions_data['A_AM_DAT_NOTIME'] = group_curr_propositions_data['A_AM_DAT'].dt.strftime('%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_NOTIME'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_NOTIME'], format='%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] = group_curr_propositions_data['A_AM_DAT_REP'].dt.strftime('%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_REP_NOTIME'], format='%d-%m-%Y')

            # NOTE: may happen that # of candidates in ranking > # candidates that were proposed to by f (with gaps in call order propositions):
            # So, lets exclude all candidates who was not proposed to, and whos CO rank is smaller than the highest CO rank among all proposed candidates

            group_curr_propositions_data[['Boursier', 'Resident']] = None
            # copying b-trait
            mapping = preferences_f.set_index('G_CN_COD')['Boursier']
            group_curr_propositions_data['Boursier'] = group_curr_propositions_data['G_CN_COD'].map(mapping)
            # copying r-trait
            mapping = preferences_f.set_index('G_CN_COD')['Resident']
            group_curr_propositions_data['Resident'] = group_curr_propositions_data['G_CN_COD'].map(mapping)

            reference_day = pd.to_datetime('2022-01-01') # January 1st 2022
            group_curr_propositions_data[['A_AM_DAT_NOTIME_DAYS_FROM_JAN1', 'A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1']] = None
            group_curr_propositions_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] = (group_curr_propositions_data['A_AM_DAT_NOTIME'] - reference_day).dt.days
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1'] = (group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] - reference_day).dt.days

            group_curr_propositions_data = group_curr_propositions_data.sort_values(by=['A_AM_DAT_NOTIME_DAYS_FROM_JAN1','Rank_CO'])
            group_curr_propositions_data = group_curr_propositions_data.reset_index(drop=True)

            propositions_data_b_only = pd.concat([propositions_data_b_only,group_curr_propositions_data], ignore_index = True) # adding to common table
            preferences_data_b_only = pd.concat([preferences_data_b_only,preferences_f], ignore_index = True) # adding to common table


    # SAVING ranks and propositions for all good b-only groups:
    with open("groups_b_only_ranks_propositions.pkl", "wb") as f:
        pickle.dump((preferences_data_b_only, propositions_data_b_only), f)

    counter_group = 0 # in total: ? groups
    number_of_internats_br = 0 # 0 internats
    propositions_data_br = pd.DataFrame(columns = ['C_GP_COD','G_CN_COD','A_SV_COD','A_AM_DAT','A_AM_DAT_REP','Rank', 'Rank_CO','A_AM_DAT_NOTIME', 'A_AM_DAT_REP_NOTIME','Boursier', 'Resident','A_AM_DAT_NOTIME_DAYS_FROM_JAN1', 'A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1'])
    preferences_data_br = pd.DataFrame(columns = ['C_GP_COD','G_CN_COD','C_CG_RAN', 'C_CG_ORD_APP','Boursier', 'Resident'])
    for f in groups_br_same: # for each good br group
        counter_group +=1
        print(f"=================== Good br group {f} ({counter_group} / {len(groups_br_same)}) ==========================")
        G_TI_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TI_COD'].item()
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()

        if G_TA_COD_curr in internats_all['G_TA_COD'].values: # if this group is for internat
            number_of_internats_br +=1
            print("It is an internat!")
        else:
            q = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item()
            share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
            share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()

            # Extracting initial ranking and call order for f:
            group_curr_rank_data = c_can_grp_processed[(c_can_grp_processed['C_GP_COD'] == f)].copy() # 'C_CG_RAN' - initial ranking, 'C_CG_ORD_APP' - call order, 'C_CG_ORD_APP_AFF' - final call order (on web-site)

            candidates_f = group_curr_rank_data['G_CN_COD'] # all candidates from ranking of f
            candidates_f = candidates_f.reset_index(drop=True)

            print(f"# Seats = {int(q)}, # Candidates = {int(candidates_f.shape[0])}")
            print(f" Quotas: for b = {share_b}%, for r = {share_r}%")

            preferences_f = group_curr_rank_data[['C_GP_COD','G_CN_COD', 'C_CG_RAN', 'C_CG_ORD_APP']].copy()

            # For each relevant candidate: add b and r indicators:
            preferences_f[['Boursier', 'Resident']] = None
            for c in candidates_f:
                preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Boursier'] = g_can.loc[g_can['G_CN_COD'] == c, 'G_CN_BRS'].item() # seems like 2 has no effect on call order (no boost)
                preferences_f.loc[preferences_f['G_CN_COD'] == c, 'Resident'] = i_ins.loc[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'] == G_TI_COD_curr), 'I_IS_FLC_SEC'].item()

            preferences_f['Boursier'] = preferences_f['Boursier'].replace(2,0) # deleting 2s from b trait
            preferences_f['Boursier'] = preferences_f['Boursier'].fillna(0) # changing all nans to 0 for b
            preferences_f['Resident'] = preferences_f['Resident'].fillna(0) # changing all nans to 0 for r
            preferences_f = preferences_f.sort_values(by='C_CG_RAN')
            preferences_f = preferences_f.reset_index(drop=True)

            # Extracting all propositions that the current group f made throughout the first stage (a_adm_cleared['A_TA_COD'] == 1) of Parcoursup
            group_curr_propositions_data_raw = a_adm_cleared[(a_adm_cleared['A_TA_COD'] == 1) & (a_adm_cleared['C_GP_COD'] == f)]

            # NOTE: no proposition in the first phase has nan in the responce date, ALSO no proposition in the first phase has A_SV_COD = -91, or -90, or -92, or 8 (cannot be proposed to type of agents)
            # 89 propositions have A_SV_COD = -39 (Fin procedure)

            # -------- Creating modified df with propositions data:
            group_curr_propositions_data = group_curr_propositions_data_raw[['C_GP_COD','G_CN_COD','A_SV_COD','A_AM_DAT','A_AM_DAT_REP']].copy()
            group_curr_propositions_data[['Rank', 'Rank_CO']] = None
            # copying ranking
            mapping = preferences_f.set_index('G_CN_COD')['C_CG_RAN']
            group_curr_propositions_data['Rank'] = group_curr_propositions_data['G_CN_COD'].map(mapping)
            # copying call order
            mapping = preferences_f.set_index('G_CN_COD')['C_CG_ORD_APP']
            group_curr_propositions_data['Rank_CO'] = group_curr_propositions_data['G_CN_COD'].map(mapping)

            # changing Status (A_SV_COD) for one of two states: accepted (1) or rejected (-1):
            group_curr_propositions_data.loc[group_curr_propositions_data['A_SV_COD'].isin([5,6,10]), 'A_SV_COD'] = 1
            group_curr_propositions_data.loc[group_curr_propositions_data['A_SV_COD'].isin([-31,-93,-50,-51,-53,-40,-41,-42,-30,-34,-25,-20,-21,-23,-24,-22,-43,-35,-36,-26,-18,-19,-27,-29,-28,-33,-94,-32,-39]), 'A_SV_COD'] = -1
            # what to do with A_SV_COD = -39 (Fin procedure)? lets put it into rejected

            group_curr_propositions_data['A_AM_DAT'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT'], format='%d%b%Y:%H:%M:%S')
            group_curr_propositions_data['A_AM_DAT_REP'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_REP'], format='%d%b%Y:%H:%M:%S')

            group_curr_propositions_data[['A_AM_DAT_NOTIME', 'A_AM_DAT_REP_NOTIME']] = None
            group_curr_propositions_data['A_AM_DAT_NOTIME'] = group_curr_propositions_data['A_AM_DAT'].dt.strftime('%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_NOTIME'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_NOTIME'], format='%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] = group_curr_propositions_data['A_AM_DAT_REP'].dt.strftime('%d-%m-%Y')
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] = pd.to_datetime(group_curr_propositions_data['A_AM_DAT_REP_NOTIME'], format='%d-%m-%Y')

            # NOTE: may happen that # of candidates in ranking > # candidates that were proposed to by f (with gaps in call order propositions):
            # So, lets exclude all candidates who was not proposed to, and whos CO rank is smaller than the highest CO rank among all proposed candidates

            group_curr_propositions_data[['Boursier', 'Resident']] = None
            # copying b-trait
            mapping = preferences_f.set_index('G_CN_COD')['Boursier']
            group_curr_propositions_data['Boursier'] = group_curr_propositions_data['G_CN_COD'].map(mapping)
            # copying r-trait
            mapping = preferences_f.set_index('G_CN_COD')['Resident']
            group_curr_propositions_data['Resident'] = group_curr_propositions_data['G_CN_COD'].map(mapping)

            reference_day = pd.to_datetime('2022-01-01') # January 1st 2022
            group_curr_propositions_data[['A_AM_DAT_NOTIME_DAYS_FROM_JAN1', 'A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1']] = None
            group_curr_propositions_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] = (group_curr_propositions_data['A_AM_DAT_NOTIME'] - reference_day).dt.days
            group_curr_propositions_data['A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1'] = (group_curr_propositions_data['A_AM_DAT_REP_NOTIME'] - reference_day).dt.days

            group_curr_propositions_data = group_curr_propositions_data.sort_values(by=['A_AM_DAT_NOTIME_DAYS_FROM_JAN1','Rank_CO'])
            group_curr_propositions_data = group_curr_propositions_data.reset_index(drop=True)

            propositions_data_br = pd.concat([propositions_data_br,group_curr_propositions_data], ignore_index = True) # adding to common table
            preferences_data_br = pd.concat([preferences_data_br,preferences_f], ignore_index = True) # adding to common table


    # SAVING ranks and propositions for all good br groups:
    with open("groups_br_ranks_propositions.pkl", "wb") as f:
        pickle.dump((preferences_data_br, propositions_data_br), f)

# LOADING ranks and propositions for all good b-only groups:
with open("groups_b_only_ranks_propositions.pkl", "rb") as f:
    preferences_data_b_only, propositions_data_b_only = pickle.load(f)

# LOADING ranks and propositions for all good br groups:
with open("groups_br_ranks_propositions.pkl", "rb") as f:
    preferences_data_br, propositions_data_br = pickle.load(f)

# for all good groups we calculate # of (stage) blocking candidates for each day (A_AM_DAT for this day and this group)

# First and last propositions (days from Jan 1):
prop_first_day_b_only = min(propositions_data_b_only['A_AM_DAT_NOTIME_DAYS_FROM_JAN1']) # 152
prop_last_day_b_only = max(propositions_data_b_only['A_AM_DAT_NOTIME_DAYS_FROM_JAN1']) # 152
prop_first_day_br = min(propositions_data_br['A_AM_DAT_NOTIME_DAYS_FROM_JAN1']) # 241
prop_last_day_br = max(propositions_data_br['A_AM_DAT_NOTIME_DAYS_FROM_JAN1']) # 241

# Last responces (days from Jan 1):
resp_last_day_b_only = max(propositions_data_b_only['A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1']) # 300
resp_last_day_br = max(propositions_data_br['A_AM_DAT_REP_NOTIME_DAYS_FROM_JAN1']) #299

# right after each block of propositions (at time t) there should be three sets of candidates for a given group f:
# (P(f,t)) with a proposition from this group
# (R(f,t)) already rejected proposition from this group
# (W(f,t)) waiting for a proposition from this group
# Then: calculate, how many candidates from W(f,t) j.e. someone from P(f,t) for a given (f,t) -- # of blocking candidates
# Then: calculate, how many candidates from W(f,t) j.e. someone from P(f,t)\(P(f,start) U ... U P(f,t-1)) for a given (f,t) (proposed at exactly day t) -- # of stage blocking candidates
# NOTE: sometimes group skips some candidates from call list while making next propositions. What to do with such candidates?:
# 1) exclude (add to R(f,start)) all candidates not present in propositions from f with call rank below the highest call rank f proposed to

# For b-only groups: -----------------------------------------------

def is_envy(c, c2, P_curr, pref_data, share_b, share_r, share_b_curr, share_r_curr): # does c have a j.e. towards c2?
    candidates_with_b_all = pref_data[pref_data['Boursier'] == 1]
    candidates_with_b_all = list(candidates_with_b_all['G_CN_COD']) # all candidates with b trait
    candidates_with_r_all = pref_data[pref_data['Resident'] == 1]
    candidates_with_r_all = list(candidates_with_r_all['G_CN_COD']) # all candidates with b trait

    # What share of proposed candidates (from P) has trait b:
    candidates_with_b_proposed = pref_data[(pref_data['G_CN_COD'].isin(candidates_with_b_all)) & (pref_data['G_CN_COD'].isin(P_curr))]
    candidates_with_b_proposed = list(candidates_with_b_proposed['G_CN_COD'])
    share_b_curr = len(candidates_with_b_proposed)/len(P_curr)

    # What share of proposed candidates (from P) has trait r:
    candidates_with_r_proposed = pref_data[pref_data['G_CN_COD'].isin(candidates_with_r_all) & (pref_data['G_CN_COD'].isin(P_curr))]
    candidates_with_r_proposed = list(candidates_with_r_proposed['G_CN_COD'])
    share_r_curr = len(candidates_with_r_proposed)/len(P_curr)

    # Ranks of c and c2 in initial ranking:
    rank_c = pref_data[pref_data['G_CN_COD'] == c]
    rank_c = int(rank_c['C_CG_RAN'].item())
    rank_c2 = pref_data[pref_data['G_CN_COD'] == c2]
    rank_c2 = int(rank_c2['C_CG_RAN'].item())

    if c in candidates_with_b_all:
        c_has_b = True
    else:
        c_has_b = False

    if c in candidates_with_r_all:
        c_has_r = True
    else:
        c_has_r = False

    if c2 in candidates_with_b_all:
        c2_has_b = True
    else:
        c2_has_b = False

    if c2 in candidates_with_r_all:
        c2_has_r = True
    else:
        c2_has_r = False

    # what happens to shares of proposed candidates with b and/or r if we exclude c2 and include c:
    share_b_new = (len(candidates_with_b_proposed) - int(c2_has_b) + int(c_has_b))/len(P_curr)
    share_r_new = (len(candidates_with_r_proposed) - int(c2_has_r) + int(c_has_r))/len(P_curr)
    delta_b_curr = max(0, share_b - share_b_curr)
    delta_r_curr = max(0, share_r - share_r_curr)
    delta_b_new = max(0, share_b - share_b_new)
    delta_r_new = max(0, share_r - share_r_new)

    if (delta_b_new < delta_b_curr) | ((delta_b_new == delta_b_curr) & (delta_r_new < delta_r_curr)) | ((delta_b_new == delta_b_curr) & (delta_r_new == delta_r_curr) & (rank_c < rank_c2)):
        return True
    else:
        return False

def number_bl_cand(W_curr, P_curr, P_now_curr, pref_data, share_b, share_r, stage_bl):
    number_bl = 0
    candidates_with_b_all = pref_data[pref_data['Boursier'] == 1]
    candidates_with_b_all = list(candidates_with_b_all['G_CN_COD']) # all candidates with b trait
    candidates_with_r_all = pref_data[pref_data['Resident'] == 1]
    candidates_with_r_all = list(candidates_with_r_all['G_CN_COD']) # all candidates with b trait

    # What share of proposed candidates (from P) has trait b:
    candidates_with_b_proposed = pref_data[(pref_data['G_CN_COD'].isin(candidates_with_b_all)) & (pref_data['G_CN_COD'].isin(P_curr))]
    candidates_with_b_proposed = list(candidates_with_b_proposed['G_CN_COD'])
    share_b_curr = len(candidates_with_b_proposed)/len(P_curr)

    # What share of proposed candidates (from P) has trait r:
    candidates_with_r_proposed = pref_data[pref_data['G_CN_COD'].isin(candidates_with_r_all) & (pref_data['G_CN_COD'].isin(P_curr))]
    candidates_with_r_proposed = list(candidates_with_r_proposed['G_CN_COD'])
    share_r_curr = len(candidates_with_r_proposed)/len(P_curr)

    for c in W_curr: # for each potentially blocking candidate
        found_blocking = False
        if stage_bl: # if looking for stage blocking
            number_cand_for_potential_envy = len(P_now_curr)
            i = 0
            while (~found_blocking) & (i < number_cand_for_potential_envy): # while stage blocking is not found and not all candidates are considered
                c2 = P_now_curr[i] # current candidate for potential j.e. from c
                found_blocking = is_envy(c, c2, P_curr, pref_data, share_b, share_r, share_b_curr, share_r_curr)
                i += 1
        else: # if looking for blocking
            number_cand_for_potential_envy = len(P_curr)
            i = 0
            while (~found_blocking) & (i < number_cand_for_potential_envy): # until blocking is not found and not all candidates are considered
                c2 = P_curr[i] # current candidate for potential j.e. from c
                found_blocking = is_envy(c, c2, P_curr, pref_data, share_b, share_r, share_b_curr, share_r_curr)
                i += 1

        if found_blocking:
            number_bl += 1

    return number_bl, share_b_curr, share_r_curr

calculate_blocking_candidates_data = False
if calculate_blocking_candidates_data:
    data_blocking_candidates_b_only = pd.DataFrame(columns = ['C_GP_COD','DAYS_FROM_JAN1','NUM_ChR_Blocking','NUM_Stage_Blocking','Share_b_curr','Share_r_curr','NUM_Not_Prop_Overall'])
    ind = 1
    for f in propositions_data_b_only['C_GP_COD'].unique(): # for each b-only group
        print(f"============== Good b-only group {f} ({ind} / {len(propositions_data_b_only['C_GP_COD'].unique())}) ==============")
        ind += 1
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()
        q = int(a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item())
        share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
        share_b = share_b/100
        share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()
        if np.isnan(share_r):
            share_r = 0
        share_r = share_r/100

        print(f" (q, share b, share r) = {q,share_b,share_r}")

        prop_data = propositions_data_b_only[propositions_data_b_only['C_GP_COD'] == f] # propositions data for this group
        prop_days = prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'].unique() # days when this group proposed
        pref_data = preferences_data_b_only[preferences_data_b_only['C_GP_COD'] == f]  # preferences data for this group
        cand_all = pref_data['G_CN_COD'].unique() # all candidates from f's ranking
        cand_prop = prop_data['G_CN_COD'].unique() # all candidates from f's propositions
        # candidates that f had not proposed to but should have:
        cand_prop_max_call_rank = int(max(prop_data['Rank_CO']))
        cand_no_prop_call_ranks = list(set(list(range(1,cand_prop_max_call_rank + 1))) - set(prop_data['Rank_CO'])) # we should not consider this candidates as ones with potential j.e.
        data_no_prop = pref_data[pref_data['C_CG_ORD_APP'].isin(cand_no_prop_call_ranks)]
        cand_no_prop = data_no_prop['G_CN_COD']
        # did f propose to the lowest call order ranked indovidual?:
        if (cand_prop_max_call_rank == int(max(pref_data['C_CG_ORD_APP']))):
            proposed_to_all = True
            number_candidates_not_proposed_overall = 0
            print(" proposed to everyone!")
        else:
            proposed_to_all = False
            number_candidates_not_proposed_overall = int(max(pref_data['C_CG_ORD_APP'])) - cand_prop_max_call_rank
            print(f" did not propose to {number_candidates_not_proposed_overall} / {int(max(pref_data['C_CG_ORD_APP'])) } candidates")

        R = [] # for lists ones who already rejected at each prop moment
        P = [] # for lists ones with active proposition at each prop moment
        P_now = [] # for lists ones with proposition made at each prop moment
        W = [] # for lists ones who waits at each prop moment

        print(f" Day    CR bl.    St. bl.    Share b    Share r")

        for t in prop_days: # for each proposition day
            prop_time = prop_data[prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] == t]
            prop_time = prop_time.reset_index(drop=True)
            prop_time = prop_time['A_AM_DAT'].iloc[0] # exact time of propositions at day t
            data_rejected = prop_data[(prop_data['A_AM_DAT_REP'] < prop_time) & (prop_data['A_SV_COD'] == -1)] # all rejected propositions at exact time of propositions at day t
            candidates_rejected = data_rejected['G_CN_COD']
            R_curr = list(set(cand_no_prop) | set(candidates_rejected)) # cand_no_prop are considered as ones who rejected f at the very beginning (at first prop moment)
            R.append(R_curr)
            data_proposed_now = prop_data[prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] == t] # all propositions made at day t
            candidates_proposed_now = data_proposed_now['G_CN_COD']
            P_now_curr = list(candidates_proposed_now)
            P_now.append(P_now_curr)
            data_proposed_before = prop_data[(prop_data['A_AM_DAT'] < prop_time) & ((prop_data['A_AM_DAT_REP'] > prop_time) | (prop_data['A_SV_COD'] == 1))] # all propositions not rejected or accepted at exact time of propositions at day t
            candidates_proposed_before = data_proposed_before['G_CN_COD']
            P_curr = list(set(candidates_proposed_before) | set(candidates_proposed_now))
            P.append(P_curr)
            W_curr = list(set(cand_all) - set(R_curr) - set(P_curr))
            number_choice_rule_blocking_candidates, share_b_curr, share_r_curr = number_bl_cand(W_curr, P_curr, P_now_curr, pref_data, share_b, share_r, False) # how many candidates from W_curr j.e. someone from P_curr
            number_stage_blocking_candidates, share_b_curr, share_r_curr = number_bl_cand(W_curr, P_curr, P_now_curr, pref_data, share_b, share_r, True) # how many candidates from W_curr j.e. someone from P_now_curr

            print(f" {t-151}      {number_choice_rule_blocking_candidates}         {number_stage_blocking_candidates}          {share_b_curr:.2f}       {share_r_curr:.2f}")

            data_blocking_candidates_curr = [f,t,number_choice_rule_blocking_candidates,number_stage_blocking_candidates,share_b_curr,share_r_curr,number_candidates_not_proposed_overall]
            new_row = pd.DataFrame([data_blocking_candidates_curr], columns = data_blocking_candidates_b_only.columns)
            data_blocking_candidates_b_only = pd.concat([data_blocking_candidates_b_only,new_row], ignore_index = True) # adding to common table

    # SAVING blocking candidates data for all good b-only groups (not internats):
    with open("groups_b_only_blocking_candidates_data.pkl", "wb") as f:
        pickle.dump(data_blocking_candidates_b_only, f)

    # For br groups: -----------------------------------------------
    data_blocking_candidates_br = pd.DataFrame(columns = ['C_GP_COD','DAYS_FROM_JAN1','NUM_ChR_Blocking','NUM_Stage_Blocking','Share_b_curr','Share_r_curr','NUM_Not_Prop_Overall'])
    ind = 1
    for f in propositions_data_br['C_GP_COD'].unique(): # for each br group
        print(f"============== Good br group {f} ({ind} / {len(propositions_data_br['C_GP_COD'].unique())}) ==============")
        ind += 1
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()
        q = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item()
        share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
        share_b = share_b/100
        share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()
        if np.isnan(share_r):
            share_r = 0
        share_r = share_r/100

        print(f" (q, share b, share r) = {q,share_b,share_r}")

        prop_data = propositions_data_br[propositions_data_br['C_GP_COD'] == f] # propositions data for this group
        prop_days = prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'].unique() # days when this group proposed
        pref_data = preferences_data_br[preferences_data_br['C_GP_COD'] == f]  # preferences data for this group
        cand_all = pref_data['G_CN_COD'].unique() # all candidates from f's ranking
        cand_prop = prop_data['G_CN_COD'].unique() # all candidates from f's propositions
        # candidates that f had not proposed to but should have:
        cand_prop_max_call_rank = int(max(prop_data['Rank_CO']))
        cand_no_prop_call_ranks = list(set(list(range(1,cand_prop_max_call_rank + 1))) - set(prop_data['Rank_CO'])) # we should not consider this candidates as ones with potential j.e.
        data_no_prop = pref_data[pref_data['C_CG_ORD_APP'].isin(cand_no_prop_call_ranks)]
        cand_no_prop = data_no_prop['G_CN_COD']

        if (cand_prop_max_call_rank == int(max(pref_data['C_CG_ORD_APP']))):
            proposed_to_all = True
            number_candidates_not_proposed = 0
            print(" proposed to everyone!")
        else:
            proposed_to_all = False
            number_candidates_not_proposed = int(max(pref_data['C_CG_ORD_APP'])) - cand_prop_max_call_rank
            print(f" did not propose to {number_candidates_not_proposed} / {int(max(pref_data['C_CG_ORD_APP'])) } candidates")

        R = [] # for lists ones who already rejected at each prop moment
        P = [] # for lists ones with active proposition at each prop moment
        P_now = [] # for lists ones with proposition made at each prop moment
        W = [] # for lists ones who waits at each prop moment

        print(f" Day    CR bl.    St. bl.    Share b    Share r")

        for t in prop_days: # for each proposition day
            prop_time = prop_data[prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] == t]
            prop_time = prop_time.reset_index(drop=True)
            prop_time = prop_time['A_AM_DAT'].iloc[0] # exact time of propositions at day t
            data_rejected = prop_data[(prop_data['A_AM_DAT_REP'] < prop_time) & (prop_data['A_SV_COD'] == -1)] # all rejected propositions at exact time of propositions at day t
            candidates_rejected = data_rejected['G_CN_COD']
            R_curr = list(set(cand_no_prop) | set(candidates_rejected)) # cand_no_prop are considered as ones who rejected f at the very beginning (at first prop moment)
            R.append(R_curr)
            data_proposed_now = prop_data[prop_data['A_AM_DAT_NOTIME_DAYS_FROM_JAN1'] == t] # all propositions made at day t
            candidates_proposed_now = data_proposed_now['G_CN_COD']
            P_now_curr = list(candidates_proposed_now)
            P_now.append(P_now_curr)
            data_proposed_before = prop_data[(prop_data['A_AM_DAT'] < prop_time) & ((prop_data['A_AM_DAT_REP'] > prop_time) | (prop_data['A_SV_COD'] == 1))] # all propositions not rejected or accepted at exact time of propositions at day t
            candidates_proposed_before = data_proposed_before['G_CN_COD']
            P_curr = list(set(candidates_proposed_before) | set(candidates_proposed_now))
            P.append(P_curr)
            W_curr = list(set(cand_all) - set(R_curr) - set(P_curr))
            number_choice_rule_blocking_candidates, share_b_curr, share_r_curr = number_bl_cand(W_curr, P_curr, P_now_curr, pref_data, share_b, share_r, False) # how many candidates from W_curr j.e. someone from P_curr
            number_stage_blocking_candidates, share_b_curr, share_r_curr = number_bl_cand(W_curr, P_curr, P_now_curr, pref_data, share_b, share_r, True) # how many candidates from W_curr j.e. someone from P_now_curr

            print(f" {t-151}      {number_choice_rule_blocking_candidates}         {number_stage_blocking_candidates}          {share_b_curr:.2f}       {share_r_curr:.2f}")

            data_blocking_candidates_curr = [f,t,number_choice_rule_blocking_candidates,number_stage_blocking_candidates,share_b_curr,share_r_curr,number_candidates_not_proposed_overall]
            new_row = pd.DataFrame([data_blocking_candidates_curr], columns = data_blocking_candidates_br.columns)
            data_blocking_candidates_br = pd.concat([data_blocking_candidates_br,new_row], ignore_index = True) # adding to common table

    # SAVING blocking candidates data for all good br groups:
    with open("groups_br_blocking_candidates_data.pkl", "wb") as f:
        pickle.dump(data_blocking_candidates_br, f)

# LOADING blocking candidates data for all good b-only groups (not internats):
with open("groups_b_only_blocking_candidates_data.pkl", "rb") as f:
    data_blocking_candidates_b_only = pickle.load(f)

# LOADING blocking candidates data for all good br groups:
with open("groups_br_blocking_candidates_data.pkl", "rb") as f:
    data_blocking_candidates_br = pickle.load(f)
#endregion

# Calculating average number of ChR-blocking and Stage-blocking candidates for resulting matching

last_stage_data_blocking_candidates_b_only = data_blocking_candidates_b_only.drop_duplicates(subset = 'C_GP_COD', keep = 'last')
avg_ChR_blocking_per_b_only = last_stage_data_blocking_candidates_b_only['NUM_ChR_Blocking'].mean()
avg_Stage_blocking_per_b_only = last_stage_data_blocking_candidates_b_only['NUM_Stage_Blocking'].mean()

last_stage_data_blocking_candidates_br = data_blocking_candidates_br.drop_duplicates(subset = 'C_GP_COD', keep = 'last')
avg_ChR_blocking_per_br = last_stage_data_blocking_candidates_br['NUM_ChR_Blocking'].mean()
avg_Stage_blocking_per_br = last_stage_data_blocking_candidates_br['NUM_Stage_Blocking'].mean()

# Segregate data for each day/stage
data_blocking_candidates_all = pd.concat([data_blocking_candidates_b_only,data_blocking_candidates_br], ignore_index = True)
first_day = min(data_blocking_candidates_all['DAYS_FROM_JAN1'])
last_day = max(data_blocking_candidates_all['DAYS_FROM_JAN1'])
numbers_of_stages = data_blocking_candidates_all['C_GP_COD'].value_counts() # numbers of stages for each good group

first_stage = 1
last_stage_all = data_blocking_candidates_all['C_GP_COD'].value_counts().max() # maximal number of stages for all
last_stage_b_only = data_blocking_candidates_b_only['C_GP_COD'].value_counts().max() # maximal number of stages for b-only
last_stage_br = data_blocking_candidates_br['C_GP_COD'].value_counts().max() # maximal number of stages for br

last_stage_avg_all = data_blocking_candidates_all['C_GP_COD'].value_counts().mean() # average of all last stages
last_stage_avg_b_only = data_blocking_candidates_b_only['C_GP_COD'].value_counts().mean() # average of b-only last stages
last_stage_avg_br = data_blocking_candidates_br['C_GP_COD'].value_counts().mean() # average of br last stages

# adding shares b and r compliances, and extending data for each group so each has the same number of stages
data_blocking_candidates_all_complete = pd.DataFrame(columns = ['C_GP_COD','DAYS_FROM_JAN1','NUM_ChR_Blocking','NUM_Stage_Blocking','Share_b_curr','Share_r_curr','NUM_Not_Prop_Overall','Share_b_compl_curr','Share_r_compl_curr','Stage'])
ind = 1
for f in data_blocking_candidates_all['C_GP_COD'].unique(): # for each group from the list
    print(f"============== Good group {f} ({ind} / {len(data_blocking_candidates_all['C_GP_COD'].unique())}) ==============")
    ind += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G_TA_COD_curr = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'G_TA_COD'].item()
        q = a_rec_grp_curr.loc[a_rec_grp_curr['C_GP_COD'] == f, 'A_RG_PLA'].item()
        share_b = a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].item()
        share_b = share_b/100
        share_r = 100 - a_rec_cleared.loc[a_rec_cleared['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].item()
        if np.isnan(share_r):
            share_r = 0
        share_r = share_r/100

        data_group_curr = data_blocking_candidates_all[data_blocking_candidates_all['C_GP_COD'] == f]

        data_group_curr.loc[:,'Share_b_compl_curr'] = np.minimum(data_group_curr['Share_b_curr'] / share_b, 1)

        if f in data_blocking_candidates_b_only['C_GP_COD'].unique(): # if it is b-only group
            data_group_curr.loc[:,'Share_r_compl_curr'] = 0
        else: # if it is br group
            if share_r == 0:
                data_group_curr.loc[:,'Share_r_compl_curr'] = 0
            else:
                data_group_curr.loc[:,'Share_r_compl_curr'] = np.minimum(data_group_curr['Share_r_curr'] / share_r, 1)

        data_group_curr.loc[:,'Stage'] = range(1, len(data_group_curr) + 1)

        if len(data_group_curr) < last_stage_all: # if we need to add more data
            #data_left_stages_curr = pd.DataFrame(columns = ['C_GP_COD','DAYS_FROM_JAN1','NUM_ChR_Blocking','NUM_Stage_Blocking','Share_b_curr','Share_r_curr','NUM_Not_Prop_Overall','Share_b_compl_curr','Share_r_compl_curr','Stage'])
            data_left_stages_curr_row = data_group_curr.tail(1) # taking the last day data
            data_left_stages_curr_row.loc[:,'NUM_Stage_Blocking'] = 0 # for all stages after no stage blocking
            data_left_stages_curr = pd.concat([data_left_stages_curr_row] * (last_stage_all-len(data_group_curr)), ignore_index=True) # duplicating last row for the rest number of stages
            data_left_stages_curr.loc[:,'Stage'] = range(len(data_group_curr) + 1, last_stage_all + 1)

        data_group_curr_all = pd.concat([data_group_curr, data_left_stages_curr], ignore_index=True) # all stages data for current group
        data_blocking_candidates_all_complete = pd.concat([data_blocking_candidates_all_complete, data_group_curr_all], ignore_index=True) # adding to main df

data_blocking_candidates_all_complete = data_blocking_candidates_all_complete.groupby('C_GP_COD', group_keys=False).head(last_stage_all)
# Calculating averages:

data_avg_blocking_candidates = pd.DataFrame(columns = ['Stage','ChR_bl_Mean_all','ChR_bl_Mean_b_only','ChR_bl_Mean_br','Stage_bl_Mean_all','Stage_bl_Mean_b_only','Stage_bl_Mean_br','Share_b_compl_Mean_all','Share_b_compl_Mean_b_only','Share_b_compl_Mean_br','Share_r_compl_Mean_br'])

for st in range(first_stage,last_stage_all + 1): # for each stage
    print(f"Stage {st}")
    # for all groups: --------------------
    data_blocking_candidates_all_curr = data_blocking_candidates_all_complete[data_blocking_candidates_all_complete['Stage'] == st] # data for current stage for all groups

    data_ChR_blocking_candidates_curr = [st,data_blocking_candidates_all_curr['NUM_ChR_Blocking'].sum()/data_blocking_candidates_all_complete['C_GP_COD'].nunique()]
    data_Stage_blocking_candidates_curr = [data_blocking_candidates_all_curr['NUM_Stage_Blocking'].sum()/data_blocking_candidates_all_complete['C_GP_COD'].nunique()]
    data_Share_b_compl_curr = [data_blocking_candidates_all_curr['Share_b_compl_curr'].sum()/data_blocking_candidates_all_complete['C_GP_COD'].nunique()]

    # for b-only groups: --------------------
    data_blocking_candidates_b_only_curr = data_blocking_candidates_all_curr[data_blocking_candidates_all_curr['C_GP_COD'].isin(data_blocking_candidates_b_only['C_GP_COD'].unique())] # data for current stage for b-only groups

    data_ChR_blocking_candidates_curr.append(data_blocking_candidates_b_only_curr['NUM_ChR_Blocking'].sum()/data_blocking_candidates_b_only['C_GP_COD'].nunique())
    data_Stage_blocking_candidates_curr.append(data_blocking_candidates_b_only_curr['NUM_Stage_Blocking'].sum()/data_blocking_candidates_b_only['C_GP_COD'].nunique())
    data_Share_b_compl_curr.append(data_blocking_candidates_b_only_curr['Share_b_compl_curr'].sum()/data_blocking_candidates_b_only['C_GP_COD'].nunique())

    # for br groups: --------------------
    data_blocking_candidates_br_curr = data_blocking_candidates_all_curr[data_blocking_candidates_all_curr['C_GP_COD'].isin(data_blocking_candidates_br['C_GP_COD'].unique())] # data for current stage for br groups

    data_ChR_blocking_candidates_curr.append(data_blocking_candidates_br_curr['NUM_ChR_Blocking'].sum()/data_blocking_candidates_br['C_GP_COD'].nunique())
    data_Stage_blocking_candidates_curr.append(data_blocking_candidates_br_curr['NUM_Stage_Blocking'].sum()/data_blocking_candidates_br['C_GP_COD'].nunique())
    data_Share_b_compl_curr.append(data_blocking_candidates_br_curr['Share_b_compl_curr'].sum()/data_blocking_candidates_br['C_GP_COD'].nunique())
    data_Share_r_compl_curr = [data_blocking_candidates_br_curr['Share_r_compl_curr'].sum()/data_blocking_candidates_br['C_GP_COD'].nunique()]

    new_row = pd.DataFrame([data_ChR_blocking_candidates_curr + data_Stage_blocking_candidates_curr + data_Share_b_compl_curr + data_Share_r_compl_curr], columns = data_avg_blocking_candidates.columns)
    data_avg_blocking_candidates = pd.concat([data_avg_blocking_candidates,new_row], ignore_index = True) # adding to common table

# SAVING avg blocking candidates data:
with open("avg_blocking_candidates_data.pkl", "wb") as f:
    pickle.dump(data_avg_blocking_candidates, f)

data_avg_blocking_candidates.to_csv("avg_blocking_candidates_data.csv", index = False)

#region Statistics mining for groups -----------------------------------------------------------------------

number_candidates_ranked = c_can_grp_no_nan_rank['G_CN_COD'].nunique() # 884392 candidates are ranked by at least one group
candidates_ranked_with_b = g_can[(g_can['G_CN_COD'].isin(c_can_grp_no_nan_rank['G_CN_COD'])) & (g_can['G_CN_BRS'] == 1)]
number_candidates_ranked_with_b = candidates_ranked_with_b.shape[0]

a_rec_grp_proposed = a_rec_grp[a_rec_grp['C_GP_COD'].isin(a_adm['C_GP_COD'])] #28649 groups participated in total
number_groups_proposed = a_rec_grp_proposed['C_GP_COD'].nunique()
a_rec_grp_proposed_internats = a_rec_grp_proposed[a_rec_grp_proposed['G_TA_COD'].isin(internats_all['G_TA_COD'])] # 625 internats
number_groups_proposed_internats = a_rec_grp_proposed_internats['C_GP_COD'].nunique()
a_rec_grp_proposed_no_internats = a_rec_grp_proposed[~a_rec_grp_proposed['G_TA_COD'].isin(internats_all['G_TA_COD'])]
number_groups_proposed_no_internats = a_rec_grp_proposed_no_internats['C_GP_COD'].nunique() # 26795 are not internats

groups_no_AA = [] # for regular institutions without AA: 6615
groups_b_only = [] # for b-only regular institutions: 18038
groups_br = [] # for  br regular institutions: 2142
groups_r_only = [] # for r-only regular institutions (is empty)
groups_with_many_G_TA_COD = [] # 521

quotas_no_AA = [] # for quotas of regular institutions without AA
quotas_b_only = [] # for quotas of b-only regular institutions
quotas_br = [] # for quotas of br regular institutions
quotas_r_only = [] # for quotas of r-only regular institutions (should be empty)

shares_b_b_only = [] # for shares b of b-only regular institutions
shares_b_br = [] # for shares b of br regular institutions

shares_r_r_only = [] # for shares r of r-only regular institutions (should be empty)
shares_r_br = [] # for shares r of br regular institutions

counter_group = 0
for f in a_rec_grp_proposed_no_internats['C_GP_COD'].unique(): # for each regular institution
    counter_group +=1
    print(f"=================== Reg. Inst. {f} ({counter_group} / {a_rec_grp_proposed_no_internats['C_GP_COD'].nunique()}) ==========================")
    G_TI_COD_curr = a_rec_grp_proposed_no_internats.loc[a_rec_grp_proposed_no_internats['C_GP_COD'] == f, 'G_TI_COD'].unique().item()
    G_TA_COD_curr = a_rec_grp_proposed_no_internats.loc[a_rec_grp_proposed_no_internats['C_GP_COD'] == f, 'G_TA_COD'].unique()
    q = a_rec_grp_proposed_no_internats.loc[a_rec_grp_proposed_no_internats['C_GP_COD'] == f, 'A_RG_PLA']
    if G_TA_COD_curr.size > 1:
        G_TA_COD_curr = G_TA_COD_curr[0]
        q = sum(q)
        groups_with_many_G_TA_COD.append(f)
    else:
        q = q.item()
    G_TA_COD_curr = G_TA_COD_curr.item()
    share_b = a_rec.loc[a_rec['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_REC'].unique().item()
    share_r = 100 - a_rec.loc[a_rec['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_REC'].unique().item()
    share_b_aut = a_rec.loc[a_rec['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_BRS_OTO'].unique().item()
    share_r_aut = 100 - a_rec.loc[a_rec['G_TA_COD'] == G_TA_COD_curr, 'A_RC_TAU_NON_RES_OTO'].unique().item()

    if ((np.nan_to_num(share_b) > 0) | (np.nan_to_num(share_r) > 0)): # if there is AA
        if ((np.nan_to_num(share_b) > 0) & (np.nan_to_num(share_r) > 0)):
            groups_br.append(f)
            quotas_br.append(q)
            shares_b_br.append(share_b)
            shares_r_br.append(share_r)
        elif (np.nan_to_num(share_b) > 0):
            groups_b_only.append(f)
            quotas_b_only.append(q)
            shares_b_b_only.append(share_b)
        else:
            groups_r_only.append(f)
            quotas_r_only.append(q)
            shares_r_r_only.append(share_r)
    else:
        groups_no_AA.append(f)
        quotas_no_AA.append(q)

seats_at_no_AA = sum(quotas_no_AA)
seats_at_b_only = sum(quotas_b_only)
seats_at_br = sum(quotas_br)

bins_shares_r_br = pd.Series(shares_r_br).value_counts().sort_index()
bins_shares_b_br = pd.Series(shares_b_br).value_counts().sort_index()
bins_shares_b_b_only = pd.Series(shares_b_b_only).value_counts().sort_index()
bins_shares_b_br = pd.Series(shares_b_br).value_counts().sort_index()
bins_shares_b_both = pd.Series(shares_b_b_only + shares_b_br).value_counts().sort_index()

# distribution of shares of residenships per candidate:
shares_distr_calc = False
if shares_distr_calc:
    residentships_share = [] # for shares of residenships for each candidate
    wishes_number = [] # for number of wishes for each candidate

    counter_cand = 0
    candidates_all = c_can_grp_no_nan_rank['G_CN_COD'].unique() # every candidate ranked by at least one group
    n=50 # how many percents of candidates to sample
    candidates_subset = np.random.choice(candidates_all, size = int(np.ceil(candidates_all.shape[0]*0.01*n)), replace=False) # too many, need to choose n% at random
    for c in candidates_subset: # for every sampled candidate
        counter_cand +=1
        print(f"=================== Candidate {c} ({counter_cand} / {candidates_subset.size}) ==========================")
        groups_curr = c_can_grp_no_nan_rank.loc[c_can_grp_no_nan_rank['G_CN_COD'] == c, 'C_GP_COD'] # all groups that ranked this candidate
        number_groups_curr = groups_curr.shape[0]
        G_TI_COD_curr = a_rec_grp_proposed.loc[a_rec_grp_proposed['C_GP_COD'].isin(groups_curr), 'G_TI_COD'].unique() # TI codes for these groups
        residentships_curr = i_ins[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'].isin(G_TI_COD_curr))] # data with r traits for these groups
        if residentships_curr.shape[0]>0:
            residentships_share_curr = residentships_curr[residentships_curr['I_IS_FLC_SEC'] == 1].shape[0]/residentships_curr.shape[0]
        else:
            residentships_share_curr = 0
        residentships_share.append(residentships_share_curr)
        wishes_number.append(residentships_curr.shape[0])
        print(f"r for share {residentships_share_curr:.2f} out of {residentships_curr.shape[0]}")
        print(f"Averages: {np.mean(residentships_share):.2f}, {np.mean(wishes_number):.2f}")

bins_residentships_share = pd.Series(residentships_share).value_counts().sort_index()
bins_wishes_number = pd.Series(wishes_number).value_counts().sort_index()

#endregion
#region Joint empirical distribution for candidates:

emp_distr_calc = False
if emp_distr_calc:
    Has_b = []
    Number_of_wishes = [] # for number of wishes for each candidate
    Number_of_wishes_internats = []
    Number_of_wishes_b_only = [] # for number of b-only wishes for each candidate
    Number_of_wishes_br = [] # for number of br wishes for each candidate
    Number_of_wishes_br_with_r = [] # for number of br wishes for each candidate, where this candidate has trait r

    counter_cand = 0
    pr = 100
    candidates_all = c_can_grp_no_nan_rank['G_CN_COD'].unique() # every candidate ranked by at least one group
    n = 100 # how many percents of candidates to sample
    if n < 100:
        candidates_subset = np.random.choice(candidates_all, size = int(np.ceil(candidates_all.shape[0]*0.01*n)), replace=False) # too many, need to choose n% at random
    else:
        candidates_subset = candidates_all

    for c in candidates_subset: # for every sampled candidate
        counter_cand +=1
        if counter_cand % pr == 0: # printing for each pr-th candidate
            print(f"=================== Candidate {c} ({counter_cand} / {candidates_subset.size}) ==========================")

        has_b_curr = g_can.loc[g_can['G_CN_COD'] == c, 'G_CN_BRS'].item()
        if has_b_curr == 1:
            Has_b.append(1)
        else:
            Has_b.append(0)

        groups_curr = c_can_grp_no_nan_rank.loc[c_can_grp_no_nan_rank['G_CN_COD'] == c, 'C_GP_COD'].tolist() # all groups that ranked this candidate
        Number_of_wishes.append(len(groups_curr))

        wishes_internats_curr = a_rec_grp_proposed_internats.loc[a_rec_grp_proposed_internats['C_GP_COD'].isin(groups_curr), 'C_GP_COD'].unique().tolist()
        Number_of_wishes_internats.append(len(wishes_internats_curr))

        wishes_b_only_curr = list(set(groups_curr) & set(groups_b_only))
        Number_of_wishes_b_only.append(len(wishes_b_only_curr))

        wishes_br_curr = list(set(groups_curr) & set(groups_br))
        Number_of_wishes_br.append(len(wishes_br_curr))

        G_TI_COD_br_curr = a_rec_grp_proposed.loc[a_rec_grp_proposed['C_GP_COD'].isin(wishes_br_curr), 'G_TI_COD'].unique() # TI codes for br wishes

        residentships_curr = i_ins[(i_ins['G_CN_COD'] == c) & (i_ins['G_TI_COD'].isin(G_TI_COD_br_curr))] # data with r traits for br wishes
        Number_of_wishes_br_with_r.append(residentships_curr[residentships_curr['I_IS_FLC_SEC'] == 1].shape[0])

    data_emp_distr_candidates = pd.DataFrame({
        'Has_b': Has_b,
        'Number_of_wishes': Number_of_wishes,
        'Number_of_wishes_b_only': Number_of_wishes_b_only,
        'Number_of_wishes_br': Number_of_wishes_br,
        'Number_of_wishes_br_with_r': Number_of_wishes_br_with_r,
        'Number_of_wishes_internats': Number_of_wishes_internats
    })

    if data_emp_distr_candidates[~(data_emp_distr_candidates['Number_of_wishes'] >= data_emp_distr_candidates['Number_of_wishes_b_only'] + data_emp_distr_candidates['Number_of_wishes_br'] + data_emp_distr_candidates['Number_of_wishes_internats'])].shape[0] > 0:
        print("There are candidates with more b-only+br wishes than total")

    if data_emp_distr_candidates[~(data_emp_distr_candidates['Number_of_wishes_br'] >= data_emp_distr_candidates['Number_of_wishes_br_with_r'])].shape[0] > 0:
        print("There are candidates with more br wishes with r than just br")

    # What to do with candidates with > 10 wishes: cut or normalize to 10? better to cut
    data_emp_distr_candidates_below11_with_internats = data_emp_distr_candidates[data_emp_distr_candidates['Number_of_wishes'] < 11]
    data_emp_distr_candidates_below11 = data_emp_distr_candidates_below11_with_internats.drop(columns='Number_of_wishes_internats')

    data_emp_distr_candidates_final = data_emp_distr_candidates_below11.groupby(data_emp_distr_candidates_below11.columns.tolist()).size().reset_index(name='Count') # creating joint empirical distribution
    data_emp_distr_candidates_final['Probability'] = data_emp_distr_candidates_final['Count'] / data_emp_distr_candidates_final['Count'].sum()
    #data_emp_distr_candidates_final = data_emp_distr_candidates_final.drop(colums = 'Count')

    # SAVING empirical distr candidates data:
    with open("emp_distr_candidates_data.pkl", "wb") as f:
        pickle.dump(data_emp_distr_candidates_final, f)

    data_emp_distr_candidates_final.to_csv("emp_distr_candidates_data.csv", index = False)

    data_emp_distr_candidates_final_with_internats = data_emp_distr_candidates_below11_with_internats.groupby(data_emp_distr_candidates_below11_with_internats.columns.tolist()).size().reset_index(name='Count') # creating joint empirical distribution taking into account internat wishes
    data_emp_distr_candidates_final_with_internats['Probability'] = data_emp_distr_candidates_final_with_internats['Count'] / data_emp_distr_candidates_final_with_internats['Count'].sum()
    #data_emp_distr_candidates_final = data_emp_distr_candidates_final.drop(colums = 'Count')

    # SAVING empirical distr candidates data:
    with open("emp_distr_candidates_data_with_internats.pkl", "wb") as f:
        pickle.dump(data_emp_distr_candidates_final_with_internats, f)

    data_emp_distr_candidates_final_with_internats.to_csv("emp_distr_candidates_data_with_internats.csv", index = False)

    # excluding all rows with less than 5 candidates (CASD request to export):
    data_emp_distr_candidates_final_with_internats_more_than_5 = data_emp_distr_candidates_final_with_internats[data_emp_distr_candidates_final_with_internats['Count']>4]
    data_emp_distr_candidates_final_with_internats_more_than_5['Probability'] = data_emp_distr_candidates_final_with_internats_more_than_5['Count'] / data_emp_distr_candidates_final_with_internats_more_than_5['Count'].sum()

    data_emp_distr_candidates_final_more_than_5 = data_emp_distr_candidates_final[data_emp_distr_candidates_final['Count']>4]
    data_emp_distr_candidates_final_more_than_5['Probability'] = data_emp_distr_candidates_final_more_than_5['Count'] / data_emp_distr_candidates_final_more_than_5['Count'].sum()

    # SAVING empirical distr candidates data:
    with open("emp_distr_candidates_data_with_internats_more_than_5.pkl", "wb") as f:
        pickle.dump(data_emp_distr_candidates_final_with_internats_more_than_5, f)

    data_emp_distr_candidates_final_with_internats_more_than_5.to_csv("emp_distr_candidates_data_with_internats_more_than_5.csv", index = False)

    # SAVING empirical distr candidates data:
    with open("emp_distr_candidates_data_more_than_5.pkl", "wb") as f:
        pickle.dump(data_emp_distr_candidates_final_more_than_5, f)

    data_emp_distr_candidates_final_more_than_5.to_csv("emp_distr_candidates_data_more_than_5.csv", index = False)
#endregion

# LOADING:
with open("emp_distr_candidates_data.pkl", "rb") as f:
    data_emp_distr_candidates_final = pickle.load(f)

 # LOADING:
with open("emp_distr_candidates_data_with_internats.pkl", "rb") as f:
    data_emp_distr_candidates_final_with_internats = pickle.load(f)