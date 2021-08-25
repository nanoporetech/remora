from torch import nn
import torch.nn.utils.rnn as rnn
import torch


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1, 16, 1)
        self.pps = rnn.pad_packed_sequence
        self.fc1 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x, x_len):
        x = self.lstm(x)

        x, hn = self.pps(x[0])
        x = x[x_len - 1]
        x = torch.transpose(torch.diagonal(x), 0, 1)
        x = self.fc1(x)
        # x = self.relu(x)

        return x
