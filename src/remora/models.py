from torch import nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch


class SimpleLSTM(nn.Module):
    def __init__(self, out_size=2):
        super().__init__()

        self.lstm = nn.LSTM(1, 32, 1)
        self.pps = rnn.pad_packed_sequence
        self.fc1 = nn.Linear(32, out_size)
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

    def forward(self, x):

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.sigmoid(self.fc2(x)))

        return x


class CNN(nn.Module):
    def __init__(self, batch_size, channel_size, out_size=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, channel_size, 8)
        self.conv2 = nn.Conv1d(32, 32, 2)
        self.fc1 = nn.Linear(32, out_size)

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # x = torch.flatten(x, start_dim=0)
        x = torch.sigmoid(self.fc1(x))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        # x = self.dropout(F.relu(self.fc4(x)))
        # x = torch.sigmoid(self.fc5(x))

        return x
