import torch
from torch import nn

import torch.nn.functional as F


class Chvon(nn.Module): # no padding
    def __init__(self, in_channels=4, out_channels=1):
        super(Chvon, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, stride=1, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=32, stride=1, padding=0)
        self.BiLSTM = nn.LSTM(input_size=54, hidden_size=40,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.Drop1 = nn.Dropout(0.2)

        # self.attention = MultiHeadAttention(160  ,4)

        self.Linear1 = nn.Linear(25600, 1024)
        self.Linear2 = nn.Linear(1024, 1)

        self.Drop2 = nn.Dropout(0.5)

    def forward(self, input):

        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)

        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)

        x, _ = self.BiLSTM(x)

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)

        x = self.Linear2(x)
        x = torch.sigmoid(x)
        return x

## bigger max pool
class Chvon2(nn.Module): # no padding
    def __init__(self, in_channels=4, out_channels=1):
        super(Chvon2, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8, stride=1, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=32, stride=1, padding=0)
        self.BiLSTM = nn.LSTM(input_size=11, hidden_size=10,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.Drop1 = nn.Dropout(0.2)

        # self.attention = MultiHeadAttention(160  ,4)

        self.Linear1 = nn.Linear(6400, 256)
        self.Linear2 = nn.Linear(256, 1)

        self.Drop2 = nn.Dropout(0.5)

    def forward(self, input):

        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)

        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)

        x, _ = self.BiLSTM(x)

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)

        x = self.Linear2(x)
        x = torch.sigmoid(x)
        return x

