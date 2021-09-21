from torch import nn
import torch.nn.utils.rnn as rnn
import torch

from remora.modules import swish
from remora import constants


class network(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=constants.DEFAULT_SIZE,
        kmer_size=constants.DEFAULT_KMER_SIZE,
        num_out=2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, size, 3)
        self.conv2 = nn.Conv2d(1, size, (kmer_size * 4, 3))
        self.conv3 = nn.Conv2d(1, 4, (size * 2, 3))

        self.fc1 = nn.Linear(kmer_size ** 2, num_out)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        sigs = sigs.permute(1, 2, 0)
        sigs = self.dropout(swish(self.conv1(sigs)))

        seqs = seqs.permute(1, 2, 0)
        seqs = self.dropout(swish(self.conv2(seqs.unsqueeze(1))))
        seqs = seqs.squeeze(2)
        z = torch.cat((sigs, seqs), 1)
        z = self.dropout(swish(self.conv3(z.unsqueeze(1))))
        z = z.squeeze(2)
        z = torch.flatten(z, start_dim=1)
        z = torch.softmax(self.fc1(z), dim=1)

        return z
