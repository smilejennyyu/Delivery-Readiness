# %%
import os
from src.s3_utils import *
from src.models import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings
import sys
import re
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import argparse
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--train_file')
parser.add_argument('--test_file')
parser.add_argument('--output_val_file')
parser.add_argument('--output_test_file')

args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file
output_val_file = args.output_val_file
output_test_file = args.output_test_file
# %%
with open(train_file, 'rb') as handle:
    train = pickle.load(handle)
with open(test_file, 'rb') as handle:
    test = pickle.load(handle)
fold_i = re.findall(r'\d+', train_file)
fold_i = int(fold_i[0])
# %%
indices = list(range(len(train['y'])))

# X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(train['X'], train['y'], indices, test_size=0.1, random_state=0) #(sample size, #feature, #time steps)
X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(train['X'], train['y'], indices, test_size=0.1, shuffle=False) #(sample size, #feature, #time steps)
X_train_scaled = np.zeros_like(X_train)
X_val_scaled = np.zeros_like(X_val)
X_test = test['X']
X_test_scaled = np.zeros_like(X_test)
y_test = np.array(test['y'])

for i in range(X_train.shape[1]):
    scaler = MinMaxScaler().fit(train['X'][:,i,:])
    X_train_scaled[:,i,:] = scaler.transform(X_train[:,i,:])
    X_val_scaled[:,i,:] = scaler.transform(X_val[:,i,:])
    X_test_scaled[:,i,:] = scaler.transform(X_test[:,i,:])
# %%
#Flipping so it goes batchsize, seq len, num features
X_train_scaled = torch.flip(torch.tensor(X_train, dtype=torch.float), [1,2])
y_train = torch.tensor(y_train, dtype=torch.float)
X_val_scaled = torch.flip(torch.tensor(X_val, dtype=torch.float), [1,2])
y_val = torch.tensor(y_val, dtype=torch.float)
X_test_scaled = torch.flip(torch.tensor(X_test, dtype=torch.float), [1,2])
y_test = torch.tensor(y_test, dtype=torch.float)
train_dataset= TensorDataset(X_train_scaled, y_train)
val_dataset= TensorDataset(X_val_scaled, y_val)
test_dataset= TensorDataset(X_test_scaled, y_test)
print(f'X_train: {X_train.shape}')

batch_size = 32
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size
        )
val_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset),
            batch_size = len(val_dataset)
        )
test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset),
            batch_size = len(test_dataset)
        )

# %%
model = LSTMClassifier(X_train.shape[2], #num features 
                1, #num classes,
                20, #hidden size
                rnn="lstm" #rnn type
           )
model.to(device)
fig_filename = output_val_file.replace('.xlsx', '.png')
model, history, output_val, y_pred_val = train_model(model, train_dataloader, 3000, 1e-6, fig_filename, val_dataloader=val_dataloader, fold=fold_i)
# %%
confidence = []
for i in output_val:
    if i[0] < 0.5:
        confidence.append(1 - i[0])
    else:
        confidence.append(i[0])
print(f'AUC score: {roc_auc_score(y_val, output_val.T[0])}.')
#%%
days_before_delivery = ''
if 'readiness_date' in test:
    days_before_delivery = 'readiness_date'
if 'start_date' in test:
    days_before_delivery = 'start_date'
# %%
results = pd.DataFrame({
    'user_id': np.array(train['uid'])[indices_val].tolist(),
    'ground_truth': y_val.tolist(),
    'prediction': y_pred_val.T[0].tolist(),
    'correctness': (y_val.numpy() == y_pred_val.T[0]).tolist(),
    'y_score': output_val.T[0].tolist(),
    'confidence': confidence,
    'x_number_of_days_before_delivery': np.array(train[days_before_delivery])[indices_val].tolist()
})
# %%
results.to_excel(output_val_file)  
# y_pred_train, output_train = evaluate_model(model, X_train_scaled, y_train)

# # %%
y_pred_test, output_test = evaluate_model(model, X_test_scaled, y_test, fold=fold_i)
confidence = []
for i in output_test:
    if i[0] < 0.5:
        confidence.append(1 - i[0])
    else:
        confidence.append(i[0])
# %%
results = pd.DataFrame({
    'user_id': test['uid'],
    'ground_truth': y_test.tolist(),
    'prediction': y_pred_test.T[0].tolist(),
    'correctness': (y_test.numpy() == y_pred_test.T[0]).tolist(),
    'y_score': output_test.T[0].tolist(),
    'confidence': confidence,
    'x_number_of_days_before_delivery': test[days_before_delivery]
})
# %%
results.to_excel(output_test_file)