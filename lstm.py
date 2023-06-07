# %%
import os
from src.s3_utils import *
from src.models import *
from sklearn.preprocessing import StandardScaler
import warnings
import sys
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
parser.add_argument('--output_file')

args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file
output_file = args.output_file
# %%
with open(train_file, 'rb') as handle:
    train = pickle.load(handle)
with open(test_file, 'rb') as handle:
    test = pickle.load(handle)
# %%
X_train = train['X'] #(sample size, #feature, #time steps)
X_train_scaled = np.zeros_like(X_train)
y_train = np.array(train['y'])
X_test = test['X']
X_test_scaled = np.zeros_like(X_test)
y_test = np.array(test['y'])

for i in range(X_train.shape[1]):
    scaler = StandardScaler().fit(X_train[:,i,:])
    X_train_scaled[:,i,:] = scaler.transform(X_train[:,i,:])
    X_test_scaled[:,i,:] = scaler.transform(X_test[:,i,:])
# %%
#Flipping so it goes batchsize, seq len, num features
X_train_scaled = torch.flip(torch.tensor(X_train, dtype=torch.float), [1,2])
y_train = torch.tensor(y_train, dtype=torch.float)
X_test_scaled = torch.flip(torch.tensor(X_test, dtype=torch.float), [1,2])
y_test = torch.tensor(y_test, dtype=torch.float)
train_dataset= TensorDataset(X_train_scaled, y_train)
test_dataset= TensorDataset(X_test_scaled, y_test)

batch_size = 16
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size
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
fig_filename = output_file.replace('.xlsx', '.png')
model, history, output_test, y_pred = train_model(model, train_dataloader, 3000, 1e-3, fig_filename, val_dataloader=test_dataloader)
# %%
confidence = []
for i in output_test:
    if i[0] < 0.5:
        confidence.append(1 - i[0])
    else:
        confidence.append(i[0])
print(f'AUC score: {roc_auc_score(y_test, output_test.T[0])}.')
# %%
days_before_delivery = ''
if 'readiness_date' in test:
    days_before_delivery = 'readiness_date'
if 'start_date' in test:
    days_before_delivery = 'start_date'
results = pd.DataFrame({
    'user_id': test['uid'],
    'ground_truth': y_test.tolist(),
    'prediction': y_pred.T[0].tolist(),
    'correctness': (y_test.numpy() == y_pred.T[0]).tolist(),
    'y_score': output_test.T[0].tolist(),
    'confidence': confidence,
    'x_number_of_days_before_delivery': test[days_before_delivery]
})
# %%
results.to_excel(output_file)
