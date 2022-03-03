import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        
        # feel free to add functions in the class if needed
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.lstm1 = nn.LSTMCell(input_size,hidden_size,num_layers)
        self.dense = nn.Linear(self.hidden_size,self.input_size)


    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        self.timesteps = x.shape[1]
        self.h_0 = torch.randn(x.shape[0],self.hidden_size).to(self.device)
        self.c_0 = torch.randn(x.shape[0],self.hidden_size).to(self.device)
        out_final = []
        for i in range(self.timesteps):
            self.h_0,self.c_0 = self.lstm1(x[:,i,:],(self.h_0,self.c_0))
            out = self.dense(self.h_0)
            out_final.append(out)
        out_final = torch.stack(out_final,dim=1)
        # define your feedforward pass
        return out_final


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        pred = x
        c_1 = self.c_0
        h_1 = self.h_0
        out = []
        # define your feedforward pass
        for i in range(self.timesteps):
            h_1,c_1 = self.lstm1(pred,(h_1,c_1))
            pred = self.dense(h_1)
            out.append(pred)
        out = torch.stack(out,dim=1)
        return out
