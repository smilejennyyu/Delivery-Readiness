#%%
import seaborn as sns

df = sns.load_dataset("titanic").head(25)
sns.violinplot(y="class", 
                x="age", 
                data=df)
sns.swarmplot(y="class", 
                x="age", 
                data=df,
              color="white", edgecolor="gray")
# sns.stripplot(y="class", 
#                 x="age", 
#                 data=df,
#               color="white", edgecolor="gray")

# %%
import pandas as pd
import pickle
with open('/mnt/TiVaCPD/TiVaCPD/data/bump/user_0.pkl', 'rb') as f:
    input_data = pickle.load(f)
    print(input_data.shape)
with open('/mnt/TiVaCPD/TiVaCPD/out/bump/corr_score_0.pkl', 'rb') as f:
    corr_score = pickle.load(f)
    print(corr_score.shape)
with open('/mnt/TiVaCPD/TiVaCPD/out/bump/mmd_score_0.pkl', 'rb') as f:
    mmd_score = pickle.load(f)
    print(mmd_score.shape)
with open('/mnt/TiVaCPD/TiVaCPD/out/bump/series_0.pkl', 'rb') as f:
    series = pickle.load(f)
    print(series.shape)
with open('/mnt/TiVaCPD/TiVaCPD/out/bump/CorrScore_interpretability_0.pkl', 'rb') as f:
    corr_score_matrix = pickle.load(f)
    print(corr_score_matrix.shape)
# %%
import matplotlib.pyplot as plt
plt.plot(corr_score)
# %%
import os
from scipy.signal import find_peaks
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.DataFrame(columns=['correlation_type', 'cp_day', 'correlation_value'])
for i in range(26):
    suffix = f'_{i}'
    path = '/mnt/TiVaCPD/TiVaCPD/out/bump_non_scheduled/'
    corr_score_matrix_name = f'{path}CorrScore_interpretability{suffix}.pkl'
    corr_score_name = f'{path}corr_score{suffix}.pkl'
    if os.path.exists(corr_score_matrix_name):
        with open(corr_score_matrix_name, 'rb') as f:
            corr_score_matrix = pickle.load(f)
            last_peak_day = corr_score_matrix[77:87].max(axis=1).idxmax()
            last_peak_corr = corr_score_matrix[77:87].max(axis=0).idxmax()
            last_peak_corr_val = corr_score_matrix[77:87].max(axis=0).max()
            
        with open(corr_score_name, 'rb') as f:
            corr_score = pickle.load(f)
            peaks, _ = find_peaks(corr_score)
            last_peak_day_2 = 90 - peaks[-1]
        new_row = {'correlation_type':last_peak_corr, 'cp_day':last_peak_day_2, 'correlation_value': last_peak_corr_val, 'user': i}
        df = df.append(new_row, ignore_index=True)
            # last_peak_day_2 = corr_score.shape[0] - 20 + corr_score[-20:-5].argmax()
        # if last_peak_day == last_peak_day_2:
        #     print('------------------------------')
        #     print(f'{i} is true')
        # else:
        #     print('------------------------------')
        #     print(f'{i} is false')
        #     print(last_peak_day, last_peak_day_2)
    else:
        new_row = {'correlation_type':np.nan, 'cp_day':np.nan, 'correlation_value': np.nan, 'user': i}
        df = df.append(new_row, ignore_index=True)


# %%
sns.violinplot(y="correlation_value", 
                x="correlation_type", 
                data=df)
sns.swarmplot(y="correlation_value", 
                x="correlation_type", 
                data=df,
              color="white", edgecolor="gray")


# %%
data = df['cp_day'].to_numpy()
figure = plt.figure(figsize=(6,9), dpi=100);    
graph = figure.add_subplot(111);

freq = pd.value_counts(data, dropna=False)
bins = freq.index
x=graph.bar(bins, freq.values) #gives the graph without NaN

graphmissing = figure.add_subplot(111)
min_val = df['cp_day'].dropna().min()
y = graphmissing.bar([min_val-1], freq[np.nan]) #gives a bar for the number of missing values at x=0
bars = list(np.concatenate((np.array(['np.nan']), np.arange(min_val, bins.max() + 1).astype(np.int))))
x_pos = np.arange(len(bars)) + min_val - 1
plt.xticks(x_pos, bars)
plt.xlabel('birth_date - last change point date')
plt.ylabel('number of people')
figure.show()

# %%
oura_features = [
    'hr_average',
    'rmssd',
    'breath_average',
    'score',
]
with open('/mnt/TiVaCPD/TiVaCPD/out/bump_non_scheduled/series_3.pkl', 'rb') as f:
    series = pickle.load(f)
    plt.rcParams["figure.figsize"] = (20,3)
    for i in range(series.shape[1]):
        plt.plot(series[:,i], label=oura_features[i])
    plt.axvline(x=90-13, color='black', ls='--', label='change point')
    plt.axvline(x=90, color='pink', ls='--', label='birth date')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    

# %%
