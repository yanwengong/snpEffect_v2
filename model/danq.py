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
        self.Drop1 = nn.Dropout(0.2)
        self.BiLSTM = nn.LSTM(input_size=75, hidden_size=320, num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Drop2 = nn.Dropout(0.5)
        self.Linear1 = nn.Linear(320*320*2, 925) # 204800
        self.Linear2 = nn.Linear(925, self.n_class)

    def forward(self, input):
        x = self.Conv(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        #print(f'input reshaped for LSTM: {x.shape}')
        x, _ = self.BiLSTM(x)
        x = self.Drop2(x)
        #print(f'output shape from LSTM layer: {x.shape}')
        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = torch.sigmoid(x)

        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(DanQ, (4, 1000))


class Simple_DanQ(torch.nn.Module):
    def __init__(self, n_class):
        super(Simple_DanQ, self).__init__()
        self.n_class = n_class
        self.Conv = nn.Conv1d(in_channels=4,
                              out_channels=32,
                              kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=13)
        self.Drop = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=75, hidden_size=32,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(32*32*2, 32)
        self.Linear2 = nn.Linear(32, self.n_class)


    def forward(self, input):
        x = self.Conv(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        x, _ = self.BiLSTM(x)
        #print(f'output shape from LSTM layer: {x.shape}')
        x = torch.flatten(x, 1)
        #print(f'output shape from flatten layer: {x.shape}')
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = torch.sigmoid(x)

        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Simple_DanQ, (4, 1000))



class Complex_DanQ(torch.nn.Module):
    def __init__(self, n_class):
        super(Complex_DanQ, self).__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4,
                               out_channels=320, # more
                               kernel_size=30) # smaller 16, 24
        self.Conv2 = nn.Conv1d(in_channels=320,
                               out_channels=160, # smaller
                               kernel_size=12)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=11)
        self.Drop1 = nn.Dropout(0.1)
        self.BiLSTM = nn.LSTM(input_size=87, hidden_size=40,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)
        self.Linear1 = nn.Linear(12800, 256)
        self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, self.n_class)


        self.Drop2 = nn.Dropout(0.2)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)

        x = self.Drop2(x)

        x, _ = self.BiLSTM(x)
        #print(f'output shape from LSTM layer: {x.shape}')
        # x = self.attention(x,x,x)

        x = torch.flatten(x, 1)
        #print(f'output shape from flatten layer: {x.shape}')
        x = self.Linear1(x)

        x = F.relu(x)

        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        #x = torch.sigmoid(x) # TODO 03/09 change to F.sigmoid(x)
        x = F.sigmoid(x) # TODO 03/09 change to directly return
        #return F.sigmoid(x) # TODO 03/09 change to remove sigmoid
        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Complex_DanQ, (4, 1000))



class Simple_DanQ_noLSTM(torch.nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4,
                               out_channels=32,
                               kernel_size=30)
        self.Conv2 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=12)
        self.Maxpool = nn.MaxPool1d(kernel_size=13,
                                    stride=11)
        self.Drop1 = nn.Dropout(0.1)
        self.Linear1 = nn.Linear(1392, 256)
        #self.Linear2 = nn.Linear(256, 256)
        self.Linear3 = nn.Linear(256, self.n_class)


        self.Drop2 = nn.Dropout(0.2)


    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)

        x = self.Maxpool(x)

        x = self.Drop2(x)

        x = torch.flatten(x, 1)
        #print(f'output shape from flatten layer: {x.shape}')
        x = self.Linear1(x)

        x = F.relu(x)

        #x = self.Linear2(x)
        #x = F.relu(x)
        x = self.Linear3(x)
        x = F.sigmoid(x)
        return x

    def __str__(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        summary(Complex_DanQ, (4, 1000))

