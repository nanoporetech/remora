from torch import nn
from torch.nn import functional as F
import torch

from remora import constants


class network(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=constants.DEFAULT_NN_SIZE,
        kmer_len=constants.DEFAULT_KMER_LEN,
        num_out=2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, size, 8)
        self.conv2 = nn.Conv1d(size, size, 2)
        self.fc1 = nn.Linear(size, num_out)

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool1d(3)

    def forward(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        x = self.dropout(F.relu(self.conv1(sigs)))
        x = self.pool(x)
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = torch.sigmoid(self.fc1(x))

        return x
