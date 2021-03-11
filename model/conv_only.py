import torch
from torch import nn

import torch.nn.functional as F



class DeepSea(nn.Module): # no padding
    def __init__(self):
        super(DeepSea, self).__init__()

        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=30)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=13, stride=11)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=160, kernel_size=12)

        self.Drop1 = nn.Dropout(0.1)

        self.Linear1 = nn.Linear(87*160, 256)
        self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, 1)



        self.Drop2 = nn.Dropout(0.2)
        for param in self.parameters():
            print(param.data)

    def forward(self, input):

        x = self.Conv1(input)
        x = F.relu(x)


        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop2(x) # added

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)

        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        x = torch.sigmoid(x)
        return x

