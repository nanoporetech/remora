from torch import nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch

from remora import log

LOGGER = log.get_logger()

DEFAULT_SIZE = 64


################################
# Variable length input models #
################################


def swish(x):
    """Swish activation

    Swish is self-gated linear activation :math:`x sigma(x)`

    For details see: https://arxiv.org/abs/1710.05941

    Note:
        Original definition has a scaling parameter for the gating value,
        making it a generalisation of the (logistic approximation to) the GELU.
        Evidence presented, e.g. https://arxiv.org/abs/1908.08681 that swish-1
        performs comparable to tuning the parameter.

    """
    return x * torch.sigmoid(x)


class SimpleLSTM(nn.Module):
    def __init__(self, size=DEFAULT_SIZE, num_out=2):
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


#############################
# Fixed length input models #
#############################


class SimpleFWLSTM(nn.Module):
    def __init__(self, size=DEFAULT_SIZE, num_out=2):
        super().__init__()

        self.lstm = nn.LSTM(1, size, 1)
        self.fc1 = nn.Linear(size, num_out)

    def forward(self, sigs, seqs):
        x, hx = self.lstm(sigs)
        x = x[-1].permute(0, 1)
        x = self.fc1(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_shape, dropout_rate=0.3):
        super().__init__()

        if dropout_rate > 1 or dropout_rate < 0:
            raise ValueError("dropout must be between 0 and 1")

        self.dropout_rate = dropout_rate

        if not isinstance(input_shape, int):
            raise ValueError("input_shape must be an integer shape")

        self.fc1 = nn.Linear(input_shape, 50)
        self.fc2 = nn.Linear(50, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, sigs, seqs):
        x = self.dropout(F.relu(self.fc1(sigs)))
        x = self.dropout(F.sigmoid(self.fc2(x)))

        return x


class CNN(nn.Module):
    def __init__(self, size=DEFAULT_SIZE, num_out=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, size, 8)
        self.conv2 = nn.Conv1d(size, size, 2)
        self.fc1 = nn.Linear(size, num_out)

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool1d(3)

    def forward(self, sigs, seqs):
        # Tensor is stored in TBF order, but `conv` requires BFT order
        x = sigs.permute(1, 2, 0)
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = torch.sigmoid(self.fc1(x))

        return x


class double_headed_CNN(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channel_size, 3)
        self.conv2 = nn.Conv2d(1, channel_size, (12, 3))
        self.conv3 = nn.Conv2d(1, 4, (channel_size * 2, 3))

        self.fc1 = nn.Linear(144, 2)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        sigs = sigs.permute(1, 2, 0)
        sigs = self.dropout(F.relu(self.conv1(sigs)))

        seqs = seqs.permute(1, 2, 0)
        seqs = self.dropout(F.relu(self.conv2(seqs.unsqueeze(1))))
        seqs = seqs.squeeze(2)
        z = torch.cat((sigs, seqs), 1)
        z = self.dropout(F.relu(self.conv3(z.unsqueeze(1))))
        z = z.squeeze(2)
        z = torch.flatten(z, start_dim=1)
        z = torch.softmax(self.fc1(z), dim=1)

        return z


class double_headed_ConvLSTM(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5)
        self.conv2 = nn.Conv1d(8, 16, 5)
        self.conv3 = nn.Conv1d(16, channel_size, 9, 3)
        self.lstm = nn.LSTM(channel_size, channel_size, 1)

        self.conv4 = nn.Conv2d(1, 4, (12, 5))

        self.fc = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sigs, seqs):
        sigs = sigs.permute(1, 2, 0)
        sigs = swish(self.conv1(sigs))

        seqs = seqs.permute(1, 2, 0)
        seqs = self.dropout(F.relu(self.conv4(seqs.unsqueeze(1))))
        seqs = seqs.squeeze(2)

        z = torch.cat((sigs, seqs), 1)

        z = swish(self.conv2(z))
        z = swish(self.conv3(z))
        z, hz = self.lstm(z.permute(2, 0, 1))
        z = z[-1].permute(0, 1)

        z = self.fc(z)

        return z
