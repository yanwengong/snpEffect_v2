import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class DanQ(torch.nn.Module):
    def __init__(self, n_class):
        super(DanQ, self).__init__()
        self.n_class = n_class
        self.Conv = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=75, hidden_size=320, num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(320 * 320 * 2, 925)
        self.Linear2 = nn.Linear(925, self.n_class)
        self.Linear3 = nn.Linear(self.n_class, 1)

    def forward(self, input):
        x = self.Conv(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        x, _ = self.BiLSTM(x)
        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        x = torch.sigmoid(x)

        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(DanQ, (4, 1000))