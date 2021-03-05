import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F
from torch.nn import MaxPool1d


class Net(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(Net, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=30)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=160, kernel_size=12)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=11)
        self.Drop1 = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=178, hidden_size=80, num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        # self.attention = MultiHeadAttention(160  ,4)

        self.Linear1 = nn.Linear(160 * 160, 925)
        self.Linear2 = nn.Linear(925, 925)
        self.Linear3 = nn.Linear(925, 1)

        self.Drop2 = nn.Dropout(0.2)

    def forward(self, input):
        batch_size = 1

        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)

        x = self.Drop2(x)

        x, _ = self.BiLSTM(x)

        # x = self.attention(x,x,x)

        x = torch.flatten(x, 1)

        x = self.Linear1(x)

        x = F.relu(x)

        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        x = torch.sigmoid(x)
        return x


# k = torch.rand(1, 4, 2000).cuda()
# Net().cuda()(k.float())
