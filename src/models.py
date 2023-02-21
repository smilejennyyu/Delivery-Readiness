import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from pathlib import Path
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                                       nn.Dropout(0.1),
                                    #    nn.Linear(self.hidden_size, self.n_state),
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

def trim_data(original_data, win_size):
    clean_data = defaultdict(list,{ k:[] for k in oura_sleep_list + oura_activity_list + ['user_id'] })
    for i in range(len(original_data['hr_5min'])):
        hr_5min = []
        rmssd_5min = []
        hypnogram_5min = []
        class_5min = []
        for d in range(len(original_data['hr_5min'][i])):
            len_hr, len_rmssd, len_hypnogram, len_class = len(original_data['hr_5min'][i][d]), len(original_data['rmssd_5min'][i][d]), len(original_data['hypnogram_5min'][i][d]), len(original_data['class_5min'][i][d])
            min_pt = min(len_hr, len_rmssd, len_hypnogram, len_class)
            if min_pt >= win_size:
                hr_5min.extend(original_data['hr_5min'][i][d][:win_size])
                rmssd_5min.extend(original_data['rmssd_5min'][i][d][:win_size])
                hypnogram_5min.extend(original_data['hypnogram_5min'][i][d][:win_size])
                class_5min.extend(original_data['class_5min'][i][d][:win_size])
            else:
                break
        if min_pt >= win_size:
            clean_data['user_id'].append(original_data['user_id'][i])
            clean_data['hr_5min'].append(hr_5min)
            clean_data['rmssd_5min'].append(rmssd_5min)
            clean_data['hypnogram_5min'].append(hypnogram_5min)
            clean_data['class_5min'].append(class_5min)
        else:
            pass
    return clean_data