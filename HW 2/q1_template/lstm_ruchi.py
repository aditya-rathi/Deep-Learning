import torch
import torch.nn as nn
from torch.autograd import Variable


class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(FlowLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.device = device

        if self.num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0.0
    
        self.lstm = nn.LSTMCell(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bias = True,
            device = self.device )
        
        self.dense = nn.Linear(self.hidden_size,self.input_size)

    

    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass
        batch, _, _, = x.shape
        #rnn = nn.LSTMCell(self.input_size,self.hidden_size)
        input = x
        hx = torch.randn(batch, self.hidden_size).to(self.device)
        cx = torch.randn(batch, self.hidden_size).to(self.device)
        output = []
        for i in range(input.shape[1]):
            hx, cx = self.lstm(input[:,i,:], (hx, cx))
            out = self.dense(hx)
            output.append(out)
        output = torch.stack(output, dim=1)
        return output


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        # define your feedforward pass
        batch, _, = x.shape
        #rnn = nn.LSTMCell(self.num_features,self.hidden_size)
        out = x
        hx = torch.randn(batch, self.hidden_size).to(self.device)
        cx = torch.randn(batch, self.hidden_size).to(self.device)
        output = []
        for i in range(19):
            hx, cx = self.lstm(out, (hx, cx))
            out = self.dense(hx)
            output.append(out)
        output = torch.stack(output, dim=1)
        return output