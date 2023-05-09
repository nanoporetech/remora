from torch import nn
import torch

from remora.activations import swish
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
        self.sig_conv1 = nn.Conv1d(1, 4, 5)
        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_conv2 = nn.Conv1d(4, 16, 5)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3)
        self.sig_bn3 = nn.BatchNorm1d(size)

        self.seq_conv1 = nn.Conv1d(kmer_len * 4, 16, 5)
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_conv2 = nn.Conv1d(16, size, 13, 3)
        self.seq_bn2 = nn.BatchNorm1d(size)

        self.merge_conv1 = nn.Conv1d(size * 2, size, 5)
        self.merge_bn = nn.BatchNorm1d(size)
        self.lstm1 = nn.LSTM(size, size, 1)
        self.lstm2 = nn.LSTM(size, size, 1)

        self.fc = nn.Linear(size, num_out)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))

        z = torch.cat((sigs_x, seqs_x), 1)

        z = swish(self.merge_bn(self.merge_conv1(z)))
        z = z.permute(2, 0, 1)
        z = swish(self.lstm1(z)[0])
        z = torch.flip(swish(self.lstm2(torch.flip(z, (0,)))[0]), (0,))
        z = z[-1].permute(0, 1)

        z = self.fc(z)

        return z
