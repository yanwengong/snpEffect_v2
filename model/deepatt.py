import torch
from torch import nn

import torch.nn.functional as F


class DeepATT(nn.Module): # no padding
    def __init__(self, in_channels=4, out_channels=1):
        super(DeepATT, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=1024, kernel_size=30, stride=1, padding=0)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=15, stride=15, padding=0)
        self.BiLSTM = nn.LSTM(input_size=64, hidden_size=64,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.Drop1 = nn.Dropout(0.2)

        # self.attention = MultiHeadAttention(160  ,4)

        self.Linear1 = nn.Linear(131072, 925)
        self.Linear2 = nn.Linear(925, 1)


    def forward(self, input):

        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)


        x, _ = self.BiLSTM(x)

        x = self.Drop1(x)

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)

        x = self.Linear2(x)
        x = torch.sigmoid(x)
        return x
