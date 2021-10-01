from torch import nn

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

        self.lstm = nn.LSTM(1, size, 1)
        self.fc1 = nn.Linear(size, num_out)

    def forward(self, sigs, seqs):
        # inputs are BFT (batch, feature, time)
        x = sigs.permute(2, 0, 1)
        x, hx = self.lstm(x)
        x = x[-1].permute(0, 1)
        x = self.fc1(x)

        return x
