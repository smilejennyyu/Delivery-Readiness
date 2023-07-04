#%%
import numpy as np
import pandas as pd
import boto3
import json
import os
import ast
import csv
import io
from io import StringIO, BytesIO, TextIOWrapper
import gzip
from datetime import datetime, date
from s3_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import ast
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import roc_auc_score
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import sys
import itertools
warnings.filterwarnings("ignore")
# import tslearn
try:
    import torch
except: pass

def largestDivisibleByKLessThanN(N, K):
    rem = N % K
    if(rem == 0):
        return N
    else:
        return N - rem

def zScore(x, dim=0):
    if isinstance(x, pd.core.frame.DataFrame):
        return (x - np.mean(x.to_numpy(), dim, keepdims=True)) / (np.std(x.to_numpy(), dim, keepdims=True) + 1e-7)
    else:
        return (x - np.mean(x, dim, keepdims=True)) / (np.std(x, dim, keepdims=True) + 1e-7)


def nanzScore(x, dim=0):
    if isinstance(x, pd.core.frame.DataFrame):
        return (x - np.nanmean(x.to_numpy(), dim, keepdims=True)) / (np.nanstd(x.to_numpy(), dim, keepdims=True) + 1e-7)
    else:
        return (x - np.nanmean(x, dim, keepdims=True)) / (np.nanstd(x, dim, keepdims=True) + 1e-7)

def get_user(user_id, start=None, end=None):
    user_sleep = df_sleep[df_sleep.user_id == user_id]#.dropna()
    user_bp = df_bodyport[df_bodyport.user_id == user_id]#.dropna()
    
    df2 = pd.merge(user_sleep, user_bp, on="date")

    if "creation_date" in df2.columns:
        for i in range(len(df2)):
            df2["creation_date"][i] = dt.datetime.strptime(df2["creation_date"][i], '%Y-%m-%d %H:%M:%S')
    # # add walk
    # user_walk = walk[walk.user_id == user_id].dropna()
    # user_walk = user_walk[["answer_text", "date"]].rename(columns={"answer_text" : "walk"})
    # user_walk["walk"] = user_walk["walk"].astype(int)
    # df2 = pd.merge(df2, user_walk, on="date", how="outer")
    
    # # add fatigue
    # user_fatigue = fatigue[fatigue.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "fatigue"})
    # user_fatigue["fatigue"] = user_fatigue["fatigue"].astype(int)
    # df2 = pd.merge(df2, user_fatigue, on="date", how="outer")

    # # add mood
    # user_mood = mood[mood.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "mood"})
    # user_mood = user_mood[["mood", "date"]]
    # user_mood["mood"] = user_mood["mood"].astype(int)
    # df2 = pd.merge(df2, user_mood, on="date", how="outer")

    
    # # add nausea
    # user_nausea = nausea[nausea.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "nausea"})
    # user_nausea = user_nausea[["nausea", "date"]]
    # user_nausea["nausea"] = user_nausea["nausea"].astype(int)
    # df2 = pd.merge(df2, user_nausea, on="date", how="outer")
    
    # # add breath
    # user_breath = breath[breath.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "breath"})
    # user_breath = user_breath[["breath", "date"]]
    # user_breath["breath"] = user_breath["breath"].astype(int)
    # df2 = pd.merge(df2, user_breath, on="date", how="outer")
    
    # # add swollen
    # user_swollen = swollen[swollen.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "swollen"})
    # user_swollen = user_swollen[["swollen", "date"]]
    # user_swollen["swollen"] = user_swollen["swollen"].astype(int)
    # df2 = pd.merge(df2, user_swollen, on="date", how="outer")
    
    # # add remember
    # user_remember = remember[remember.user_id == user_id].dropna()[["answer_text", "date"]].rename(columns={"answer_text" : "remember"})
    # user_remember = user_remember[["remember", "date"]]
    # user_remember["remember"] = user_remember["remember"].astype(int)
    # df2 = pd.merge(df2, user_remember, on="date", how="outer")
    

    df2.set_index(df2["date"], inplace=True)
    df2.sort_index(inplace=True)
    if start:
        mask = pd.to_datetime(df2["date"]).between(start.astype(str)[0], end.astype(str)[0], inclusive=True)
        df2 = df2[mask]
    return df2


# Define s3 bucket
bucket = 'fouryouandme-study-data'

#List s3 keys
# get_matching_s3_keys(bucket, prefix='bump/') #Remove prefix to view non-BUMP data
bucket = '4youandme-study-data' # for SinC project
bucket = 'fouryouandme-study-data' # for 4YouandMe, Bump, CamCog or Bodyport project

# prefix = {study_name} or {study_name}/{source}
# sources: app_activities, bodyport, camcog, garmin, oura, redcap, rescuetime
# note camcog not accessible to bodyport (and vice a versa)
get_matching_s3_keys(bucket, prefix='bump/oura')

# Study IDs

key = 'bump/redcap/wave_4/study_ids.csv.gz'
df_studyID = pandas_from_csv_s3(bucket, key=key, compression='gzip')

# Some dataframes use 'record_id' instead of 'user_id'. 
# You'll need to match it up with df_studyID where'evidation_id' is 'user_id'
# NOTE: Very few examples of this. Birthing data is the important one

# Birthing Data
key = 'bump/redcap/wave_4/birthing_data_cohort_2_only.csv.gz'
df_birth = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_birth['date'] = pd.to_datetime(df_birth.birth_date).dt.date


df_birth = pd.merge(df_birth, df_studyID, on='record_id')
df_birth['user_id'] = df_birth.evidation_id

# There is a missing value in the birthing data. I'm removing it here
df_birth = df_birth.drop(index=50)

# Bodyport Wave 2
key = 'bump/bodyport/wave_4/bodyport.csv.gz'
df_bodyport = pandas_from_csv_s3(bucket, key=key, compression='gzip')
# OPTIONAL: Convert date format
df_bodyport['date'] = pd.to_datetime(df_bodyport.event_date).dt.date 

# Garmin Wave 2
key = 'bump/garmin/wave_4/garmin_activities.csv.gz'
df_gAct = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_gAct['date'] = pd.to_datetime(df_gAct.event_date).dt.date

key = 'bump/garmin/wave_4/garmin_dailies.csv.gz'
df_gDaily = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_gDaily['date'] = pd.to_datetime(df_gDaily.event_date).dt.date

key = 'bump/garmin/wave_4/garmin_pulse_ox.csv.gz'
df_gPulse = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_gPulse['date'] = pd.to_datetime(df_gPulse.event_date).dt.date

key = 'bump/garmin/wave_4/garmin_respiration.csv.gz'
df_gResp = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_gResp['date'] = pd.to_datetime(df_gResp.event_date).dt.date

key = 'bump/garmin/wave_4/garmin_user_metrics.csv.gz'
df_gUser = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_gUser['date'] = pd.to_datetime(df_gUser.event_date).dt.date

# Oura Wave 2
key = 'bump/oura/wave_4/oura_sleep.csv.gz'
df_sleep = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_sleep['date'] = pd.to_datetime(df_sleep.event_date).dt.date

key = 'bump/oura/wave_4/oura_activity.csv.gz'
df_activity = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_activity['date'] = pd.to_datetime(df_activity.event_date).dt.date

key = 'bump/oura/wave_4/oura_readiness.csv.gz'
df_readiness = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_readiness['date'] = pd.to_datetime(df_readiness.event_date).dt.date

# Surveys Wave 2
key = 'bump/app_activities/wave_4/surveys.csv.gz'
df_survey = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_survey['date'] = pd.to_datetime(df_survey.updated_at).dt.date

key = 'bump/app_activities/wave_4/quick_activities.csv.gz'
df_sam = pandas_from_csv_s3(bucket, key=key, compression='gzip')
df_sam['date'] = pd.to_datetime(df_sam.event_date).dt.date

# Daily Symptom Survey Questions (1-7 Likert Scale) (See Data Dictionary) 
# nausea = df_survey[df_survey['question_id'] == 203]
# fatigue = df_survey[df_survey['question_id'] == 204]
# mood = df_survey[df_survey['question_id'] == 205]
# breath = df_survey[df_survey['question_id'] == 206]
# swollen = df_survey[df_survey['question_id'] == 207]
# walk = df_survey[df_survey['question_id'] == 208]
# remember = df_survey[df_survey['question_id'] == 209]

# dfs = [df_sleep, df_bodyport, df_birth, df_gAct, df_gDaily, df_gPulse, df_gResp, df_gUser, df_activity, df_readiness, df_survey, df_sam]
# dfs = [df_sleep, df_bodyport, df_birth, df_gDaily, df_activity, df_readiness, df_survey, df_sam]
dfs = [df_sleep, df_bodyport, df_birth, df_activity, df_readiness, df_survey, df_sam]

names = []
for df in dfs:
    [names.append(i) for i in df.columns.to_list()]

feature_names = [
    'creation_date', #bodyport
    'impedance_mag_1_ohms', #bodyport
    'impedance_phase_1_degs', #bodyport
    'hr_lowest', #oura
    'impedance_ratio', 
    'sway_velocity_mm_sec',
    'peripheral_fluid', #bodyport
    'score_deep', #oura
    'temperature_deviation', #oura
    'temperature_trend_deviation', #oura
    'temperature_delta', #oura
    'hr_average', #oura
    "hr_5min",
    "rmssd_5min",
    'duration', #oura
    'rem', #oura
    'rmssd', #oura
    'heart_rate', #bodyport
    'bmi_kg_m2',
    'efficiency', #oura
    'score_alignment', #oura
    'score_rem', #oura
    'total_body_water_percent',
    'light', #oura
    'onset_latency', #oura
    'restless', #oura
    'breath_average', #oura
    'score_disturbances', #oura
    'score', #oura
    'score_efficiency', #oura
    'backend_sway_area_mm2',
    'score_latency', #oura
    'score_total', #oura
    'run_time_sec',
    'bedtime_start',
    'weight_kg',
]

outcome_names = [
    "walk",
    "fatigue",
    "mood",
    "nausea",
    "breath",
    "swollen",
    "remember",
]
#merge all data features
users = []
users_id = []
users_w_birth = []
users_id_w_birth = []
for i, user_id in tqdm(enumerate(df_sleep.user_id.unique())):
    df = get_user(user_id)
    
    df = df[feature_names + ["date"]]
    if len(df) > 10:
        users_id.append(user_id)
        users.append(df)
        if (len(df_birth.loc[df_birth.user_id == user_id]) != 0):
            users_id_w_birth.append(user_id)
            users_w_birth.append(df)
#%% sort values by missingness
col = "rmssd_5min"
date_num_list = []
for u_i in range(len(users_id_w_birth)):
    pdf = users_w_birth[u_i]
    date_num_list.append(len(pdf.date))
sorted_index = sorted(range(len(date_num_list)), key=lambda k: date_num_list[k], reverse=True)
date_num_list = [date_num_list[i] for i in sorted_index]
users_w_birth = [users_w_birth[i] for i in sorted_index]
users_id_w_birth = [users_id_w_birth[i] for i in sorted_index]
#%% plot delivery +/- days features

for user in users_id_w_birth:
# user = users_id_w_birth[1]
    try:
        user_index = users_id_w_birth.index(user)
        before_days = 14
        after_days = 14

        plt.rcParams.update({'figure.max_open_warning': 0})
        sns.set_theme(style='darkgrid')
        pdf = users_w_birth[user_index]

        if (len(df_birth.loc[df_birth.user_id == user]) != 0):

            birth = df_birth.loc[df_birth.user_id == user].reset_index()
            pdf = pdf[(pdf['date'] < birth.date[0] + dt.timedelta(days=before_days)) & (pdf['date'] > birth.date[0] - dt.timedelta(days=after_days))]
            plt.figure(figsize=(12,4))
            
            # Plot birthing data if it exists for that user
            plt.axvline(x=birth.date, color = 'y', ls='--')
            ymin, ymax = plt.gca().get_ylim()
            xmin, xmax = plt.gca().get_xlim()
            # plt.text(birth.date, ymax, birth['date'][0], fontsize=12, color='y')
            
            # Dataframe of data before birth
            after = pdf[(pdf['date'] > birth.date[0]) & (pdf['date'] < birth.date[0] + dt.timedelta(days=before_days))]
            before = pdf[(pdf['date'] < birth.date[0]) & (pdf['date'] > birth.date[0] - dt.timedelta(days=after_days))]
            after = after[['date', 'bedtime_start', col]].drop_duplicates()
            before = before[['date', 'bedtime_start', col]].drop_duplicates()

            if len(before[col]) != 0:
                avaiable_date_num = len(after['date']) + len(before['date'])
                if isinstance(before[col][0], str):
                    print(f'avaiable date num:{avaiable_date_num}')
                    before_list = np.array([])
                    after_list = np.array([])
                    for i in range(len(after[col])):
                        after[col][i] = np.array(json.loads(after[col][i]))
                        after_timesteps = np.array([dt.datetime.strptime(after.bedtime_start[i], '%Y-%m-%d %H:%M:%S') + dt.timedelta(minutes=5 * x) for x in range(len(after[col][i]))])
                        after_nonzero_indices = np.where(after[col][i] != 0)[0]
                        after[col][i] = after[col][i][after_nonzero_indices]
                        after_timesteps = after_timesteps[after_nonzero_indices]
                        after_list = np.concatenate((after_list, after[col][i]))
                        plt.plot(after_timesteps, after[col][i], color='purple', linestyle='None', markersize = 1.0, marker='o')
                        plt.errorbar(x=[after.date[i]], y=[after[col][i].mean()], yerr=[after[col][i].std()], fmt='o', linewidth=2, capsize=6, ecolor='orange', color='orange', alpha=0.75)
                    for i in range(len(before[col])):
                        before[col][i] = np.array(json.loads(before[col][i]))
                        before_timesteps = np.array([dt.datetime.strptime(before.bedtime_start[i], '%Y-%m-%d %H:%M:%S') + dt.timedelta(minutes=5 * x) for x in range(len(before[col][i]))])
                        before_nonzero_indices = np.where(before[col][i] != 0)[0]
                        before[col][i] = before[col][i][before_nonzero_indices]
                        before_timesteps = before_timesteps[before_nonzero_indices]
                        before_list = np.concatenate((before_list, before[col][i]))
                        plt.plot(before_timesteps, before[col][i], color='purple', linestyle='None', markersize = 1.0, marker='o')
                        plt.errorbar(x=[before.date[i]], y=[before[col][i].mean()], yerr=[before[col][i].std()], fmt='o', linewidth=2, capsize=6, ecolor='orange', color='orange', alpha=0.75)
                    before_avg = before_list.mean()
                    after_avg = after_list.mean()

                    max_num = np.concatenate((after_list, before_list)).max()
                else:
                    sns.scatterplot(data=pdf, x='date', y=col, ci=None, color='purple')
                    before_avg = before[col].mean()
                    after_avg = after[col].mean()
                    max_num = np.concatenate((before[col], after[col])).max()
                print('Pre-birth Average: ', before_avg)
                print('Post-birth Average: ', after_avg)
                plt.text(birth.date, max_num, birth['date'][0], fontsize=12, color='y')
                plt.hlines(y=before_avg, xmin=pdf.date[0], xmax=birth.date, color='blue', linestyles='dashdot')
                plt.hlines(y=after_avg, xmin=birth.date, xmax=pdf.date[-1], color='red', linestyles='dashdot')


                plt.xlabel(''); plt.ylabel(col)
                plt.title('User ID: ' + str(user))
                plt.savefig(f'/mnt/results/birth_date_exploration/{col}_+-{before_days}/dateNum-{avaiable_date_num}-user-{user}.jpeg', bbox_inches='tight')
                plt.show()
    except:
        pass

# %%
#TODO average all users, plot mean and std

# %%
