from torch import nn
import torch.nn.utils.rnn as rnn
import torch

from remora import constants


class network(nn.Module):
    _variable_width_possible = True

    def __init__(self, size=constants.DEFAULT_SIZE, num_out=2):
        super().__init__()

        self.lstm = nn.LSTM(1, size, 1)
        self.fc1 = nn.Linear(size, num_out)

    def forward(self, sigs, seqs, lens):
        x = self.lstm(sigs)
        x, hn = rnn.pad_packed_sequence(x[0])
        x = x[lens - 1]
        x = torch.transpose(torch.diagonal(x), 0, 1)
        x = self.fc1(x)

        return x
