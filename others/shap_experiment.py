#%%
from torch.utils.data import TensorDataset
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from pathlib import Path
import random
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
class LSTMClassifier(nn.Module):
    def __init__(self, feature_size, n_state, hidden_size, rnn="GRU", regres=True, bidirectional=False, return_all=False,
                 seed=random.seed('2021')):
        
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)

        self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=self.hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(self.hidden_size, self.n_state),
                                       nn.Sigmoid())

    def forward(self, input, past_state=None, **kwargs):
        input = input.to(self.device)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[0], self.hidden_size]).to(self.device)
        if self.rnn_type == 'GRU':
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
        else:
            return encoding.view(encoding.shape[1], -1)

import copy

def train_model(model, train_dataloader, n_epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.BCELoss().to(device)

    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    iters,iters_sub, train_acc, val_acc = [], [] ,[], []
    
    best_loss = 10000.0
    
    n=0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, n_epochs))
        print('Training...')

        total_train_loss = 0
        train_losses=[]
        model = model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = (time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input= batch[0].to(device)
            target =  batch[1].to(device)
            iters.append(n)
            optimizer.zero_grad()
            out = model(b_input)
            out = torch.transpose(out,1,0)[0]

            # target = torch.argmax(b_target, 1)
            loss = criterion(out, target)

            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            train_losses.append(loss.item())
            
            if n % 10 == 0:
                iters_sub.append(n)
                
                #train_acc.append(get_accuracy(model, train_dataloader))
                #print(get_accuracy(model, train_dataloader))
                #val_acc.append(get_accuracy(model, validation_dataloader))
            # increment the iteration number
            n += 1

        training_time = (time.time() - t0)
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))


        print("")
        print("Running Validation...")

        t0 = time.time()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        val_losses = []
        model = model.eval()

        train_loss = np.mean(train_losses)
    
        history['train'].append(train_loss)
        
        print(f'Epoch {epoch}: train loss {train_loss} ')
    plt.style.use('seaborn-white')
    plt.plot(history['train'])

    plt.title('LSTM  Training Curves')
    plt.ylabel('CE Loss')
    plt.xlabel('Epoch Number')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
    return model.eval(), history

def evaluate_model(model, x_test, y_test):
    #INFERENCE ON TEST SET
    x_test.to(device)
    y_test.to(device)
    predictions = model(x_test).round()
    print(model(x_test))
    print(predictions)
    
    y_test = y_test.numpy()
    y_pred = predictions.detach().numpy()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'accuracy: {accuracy}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
#%%

X, y = np.load('/mnt/shap_test/X.npy'), np.load('/mnt/shap_test/y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

#Flipping so it goes batchsize, seq len, num features
X_train = torch.flip(torch.tensor(X_train, dtype=torch.float), [1,2])
y_train = torch.tensor(y_train, dtype=torch.float)

X_test = torch.flip(torch.tensor(X_test, dtype=torch.float), [1,2])
y_test = torch.tensor(y_test, dtype=torch.float)

train_dataset= TensorDataset(X_train, y_train)

test_dataset= TensorDataset(X_test, y_test)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 20

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size
        )

test_dataloader = DataLoader(
            test_dataset, 
            sampler = SequentialSampler(test_dataset)
        )

model = LSTMClassifier(X_train.shape[2], #num features 
                1, #num classes,
                10, #hidden size
                rnn="LSTM" #rnn type
           )
model.to(device)
model, history = train_model(model, train_dataloader,500, 1e-3)
#%%
evaluate_model(model, X_test, y_test)
#%%
import shap
oura_features = [
    'hr_lowest',
    'hr_average',
    'rmssd',
    'score_deep',
    'temperature_deviation',
    'temperature_trend_deviation',
    'temperature_delta',
    'duration',
    'rem',
    'efficiency',
    'score_alignment',
    'score_rem',
    'light',
    'onset_latency',
    'restless',
    'breath_average',
    'score_disturbances',
    'score',
    'score_efficiency',
    'score_latency',
    'score_total'
]
# It wants gradients enabled, and uses the training set
torch.set_grad_enabled(True)
e = shap.DeepExplainer(model, Variable(X_train) )

# Get the shap values from my test data (this explainer likes tensors)
shap_values = e.shap_values(Variable(X_test))

# Plots
#shap.force_plot(explainer.expected_value, shap_values, feature_names)
#shap.dependence_plot("b1_price_avg", shap_values, data, feature_names)
shap.summary_plot(shap_values, X_test, oura_features)
# %%
torch.set_grad_enabled(False)

# Get features
train_features_df = ... # pandas dataframe
test_features_df = ... # pandas dataframe

# Define function to wrap model to transform data to tensor
f = lambda x: model_list[0]( Variable( torch.from_numpy(x) ) ).detach().numpy()

# Convert my pandas dataframe to numpy
data = test_features_df.to_numpy(dtype=np.float32)

# The explainer doesn't like tensors, hence the f function
explainer = shap.KernelExplainer(f, data)

# Get the shap values from my test data
shap_values = explainer.shap_values(data)

# Enable the plots in jupyter
shap.initjs()

feature_names = test_features_df.columns
# Plots
#shap.force_plot(explainer.expected_value, shap_values[0], feature_names)
#shap.dependence_plot("b1_price_avg", shap_values[0], data, feature_names)
shap.summary_plot(shap_values[0], data, feature_names)