import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_shape, dropout_rate=0.3):
        super().__init__()

        if dropout_rate > 1 or dropout_rate < 0:
            raise ValueError('dropout must be between 0 and 1')

        self.dropout_rate = dropout_rate

        if not isinstance(input_shape,int):
            raise ValueError('input_shape must be an integer shape')


        self.fc1 = nn.Linear(input_shape, 50)
        self.fc2 = nn.Linear(50, 1)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.sigmoid(self.fc2(x)))

        return x
