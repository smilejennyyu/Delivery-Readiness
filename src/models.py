import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score
import numpy as np
from pathlib import Path
import time
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class LSTMClassifier(nn.Module):
#     def __init__(self, feature_size, n_state, hidden_size, rnn="GRU", regres=True, bidirectional=False, return_all=False,
#                  seed=random.seed('2021')):
        
#         super(LSTMClassifier, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_state = n_state
#         self.seed = seed
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.rnn_type = rnn
#         self.regres = regres
#         self.return_all = return_all
        
#         # Input to torch LSTM should be of size (seq_len, batch, input_size)
#         if self.rnn_type == 'GRU':
#             self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)
#         else:
#             self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional, batch_first=True).to(self.device)

#         self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=self.hidden_size),
#                                        nn.ReLU(),
#                                        nn.Dropout(0.3),
#                                        nn.Linear(self.hidden_size, self.n_state),
#                                        nn.Sigmoid())

#     def forward(self, input, past_state=None, **kwargs):
#         input = input.to(self.device)
#         self.rnn.to(self.device)
#         self.regressor.to(self.device)
#         if not past_state:
#             #  hidden states: (num_layers * num_directions, batch, hidden_size)
#             past_state = torch.zeros([1, input.shape[0], self.hidden_size]).to(self.device)
#         if self.rnn_type == 'GRU':
#             all_encodings, encoding = self.rnn(input, past_state)
#         else:
#             all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        
#         if self.regres:
#             if not self.return_all:
#                 return self.regressor(encoding.view(encoding.shape[1], -1))
#             else:
#                 reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
#                 return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
#         else:
#             return encoding.view(encoding.shape[1], -1)

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
                                       nn.Dropout(0.3))
        self.last_layer2 = nn.Linear(self.hidden_size, self.n_state)
        self.last_layer = nn.Sigmoid()

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
                temp = self.regressor(encoding.view(encoding.shape[1], -1))
                temp2 = self.last_layer2(temp)
                return (self.last_layer(temp2), temp)
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1]*all_encodings.shape[0],-1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0],-1))
        else:
            return encoding.view(encoding.shape[1], -1)

def train_model(model, train_dataloader, n_epochs, lr, fig_name, val_dataloader=None, fold=0):
    best_model = None
    best_auc = 0
    best_acc = 0
    best_combine = 0
    best_embeddings = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    criterion = nn.BCELoss().to(device)
    history = dict(train=[], val=[])
    best_val_pred = None
    best_val_output = None
        
    for epoch in tqdm(range(1, n_epochs + 1)):
        total_train_loss = 0
        train_losses = []
        val_losses = []
        model = model.train()
        embeddings = None

        for step, batch in enumerate(train_dataloader):
            b_input= batch[0].to(device)
            target =  batch[1].to(device)
            optimizer.zero_grad()

            out, embeddings_train = model(b_input)
            embeddings_train = embeddings_train.detach().numpy()
            if step == 0:
                embeddings = embeddings_train
            else:
                embeddings = np.vstack((embeddings, embeddings_train))
            # embeddings = model(b_input)[1]
            out = torch.transpose(out,1,0)[0]

            loss = criterion(out, target)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            train_losses.append(loss.item())

        if val_dataloader != None:
            for batch in val_dataloader:
                val_input = batch[0].to(device)
                val_target = batch[1].to(device)
                val_predictions, embeddings_val = model(val_input)

                embeddings_val = embeddings_val.detach().numpy()
                embeddings = np.vstack((embeddings, embeddings_val))

                val_predictions_t = torch.transpose(val_predictions,1,0)[0]
                val_loss = criterion(val_predictions_t, val_target)
                val_losses.append(val_loss.item())

        model = model.eval()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        if val_dataloader != None:
            history['val'].append(val_loss)
            output_test = val_predictions.detach().numpy().T
            auc = roc_auc_score(val_target, val_predictions.detach().numpy().T[0])
            acc = accuracy_score(val_target.numpy(), val_predictions.detach().numpy().T[0].round())
            confidence = []
            for i in output_test:
                if i[0] < 0.5:
                    confidence.append(1 - i[0])
                else:
                    confidence.append(i[0])
            confidence = np.array(confidence)
            combine = confidence.mean() * 2 + acc
            if combine > best_combine:
                best_combine = combine
                best_model = model
                best_embeddings = embeddings
                best_val_pred = val_predictions.detach().numpy()
                best_val_output = best_val_pred.round()
                accuracy = accuracy_score(val_target.numpy(), best_val_output.T[0])
                f1 = f1_score(val_target.numpy(), best_val_output.T[0])
                precision = precision_score(val_target.numpy(), best_val_output.T[0])
    np.save(f'/repos/Delivery-Readiness/embeddings/set_d_train_{fold}.npy', embeddings)
    plt.style.use('seaborn-white')
    plt.plot(history['train'])
    if val_dataloader != None:
        plt.plot(history['val'])
    plt.title('LSTM  Training Curves')
    plt.ylabel('CE Loss')
    plt.xlabel('Epoch Number')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(fig_name)

    accuracy = accuracy_score(val_target.numpy(), best_val_output.T[0])
    f1 = f1_score(val_target.numpy(), best_val_output.T[0])
    precision = precision_score(val_target.numpy(), best_val_output.T[0])
    print(f'Accuracy: {accuracy}')
    print(f'F1: {f1}')
    print(f'Precision: {precision}')
    return best_model, history, best_val_pred, best_val_output

def evaluate_model(model, x_test, y_test, fold=0):
    #INFERENCE ON TEST SET
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    predictions, embeddings = model(x_test)

    predictions = predictions.detach().numpy()
    embeddings = embeddings.detach().numpy()
    np.save(f'/repos/Delivery-Readiness/embeddings/set_d_test_{fold}.npy', embeddings)
    
    y_test = y_test.numpy()
    y_pred = predictions.round()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f'accuracy: {accuracy}')
    print(f'f1: {f1}')
    print(f'precision: {precision}')
    return y_pred, predictions