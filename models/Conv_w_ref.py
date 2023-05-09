from torch import nn
import torch

from remora.activations import swish
from remora import constants
import torch.nn.functional as F


class network(nn.Module):
    _variable_width_possible = False

    def __init__(
        self,
        size=constants.DEFAULT_NN_SIZE,
        kmer_len=constants.DEFAULT_KMER_LEN,
        num_out=2,
    ):
        super().__init__()
        self.sig_conv1 = nn.Conv1d(1, 4, 11)
        self.sig_bn1 = nn.BatchNorm1d(4)
        self.sig_conv2 = nn.Conv1d(4, 16, 11)
        self.sig_bn2 = nn.BatchNorm1d(16)
        self.sig_conv3 = nn.Conv1d(16, size, 9, 3)
        self.sig_bn3 = nn.BatchNorm1d(size)

        self.seq_conv1 = nn.Conv1d(kmer_len * 4, 16, 11)
        self.seq_bn1 = nn.BatchNorm1d(16)
        self.seq_conv2 = nn.Conv1d(16, 32, 11)
        self.seq_bn2 = nn.BatchNorm1d(32)
        self.seq_conv3 = nn.Conv1d(32, size, 9, 3)
        self.seq_bn3 = nn.BatchNorm1d(size)

        self.merge_conv1 = nn.Conv1d(size * 2, size, 5)
        self.merge_bn1 = nn.BatchNorm1d(size)
        self.merge_conv2 = nn.Conv1d(size, size, 5)
        self.merge_bn2 = nn.BatchNorm1d(size)

        self.merge_conv3 = nn.Conv1d(size, size, 3, stride=2)
        self.merge_bn3 = nn.BatchNorm1d(size)
        self.merge_conv4 = nn.Conv1d(size, size, 3, stride=2)
        self.merge_bn4 = nn.BatchNorm1d(size)

        self.fc = nn.Linear(size * 3, num_out)

    def forward(self, sigs, seqs):
        sigs_x = swish(self.sig_bn1(self.sig_conv1(sigs)))
        sigs_x = swish(self.sig_bn2(self.sig_conv2(sigs_x)))
        sigs_x = swish(self.sig_bn3(self.sig_conv3(sigs_x)))

        seqs_x = swish(self.seq_bn1(self.seq_conv1(seqs)))
        seqs_x = swish(self.seq_bn2(self.seq_conv2(seqs_x)))
        seqs_x = swish(self.seq_bn3(self.seq_conv3(seqs_x)))
        z = torch.cat((sigs_x, seqs_x), 1)

        z = swish(self.merge_bn1(self.merge_conv1(z)))
        z = swish(self.merge_bn2(self.merge_conv2(z)))
        z = swish(self.merge_bn3(self.merge_conv3(z)))
        z = swish(self.merge_bn4(self.merge_conv4(z)))

        z = torch.flatten(z, start_dim=1)
        z = self.fc(z)

        return z
