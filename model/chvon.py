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

class Chvon2(nn.Module): # no padding
    def __init__(self, n_class):
        super(Chvon2, self).__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=160, kernel_size=16,
                               stride=1, padding=0) #0
        self.Maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0) #1
        self.Conv2 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=32,
                               stride=1, padding=0) #2
        self.BiLSTM = nn.LSTM(input_size=160, hidden_size=160,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True) #3

        self.Drop1 = nn.Dropout(0.2) #4

        # self.attention = MultiHeadAttention(160  ,4)

        self.Linear1 = nn.Linear(16960, 925) #5
        self.Linear2 = nn.Linear(925, self.n_class) #6
        #self.Linear3 = nn.Linear(126, self.n_class)

    def forward(self, input):

        x = self.Conv1(input) #1
        x = F.relu(x)
        x = self.Maxpool1(x) #2
        x = self.Drop1(x)

        x = self.Conv2(x) #3
        x = F.relu(x)
        x = self.Maxpool1(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)  # x_x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.BiLSTM(x_x)  # torch.Size([100, 75, 640]) #3

        x = torch.flatten(x, 1)
        x = self.Linear1(x) #4
        x = F.relu(x)

        x = self.Linear2(x) #5

        #x = torch.sigmoid(x)
        return x


## bigger max pool
class Chvon3(nn.Module): # no padding
    def __init__(self, n_class):
        super(Chvon3, self).__init__()
        self.n_class = n_class
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=160,
                              num_layers=2,
                              batch_first=True,
                              dropout=0.4,
                              bidirectional=True)

        self.Drop1 = nn.Dropout(0.2)

        # self.attention = MultiHeadAttention(160  ,4)

        # self.Linear1 = nn.Linear(272640, 800)
        # #self.Linear1 = nn.Linear(1696, 80)
        # self.Linear2 = nn.Linear(800, self.n_class)
        # #self.Linear3 = nn.Linear(126, self.n_class)

        self.Linear1 = nn.Linear(272640, 800)

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
        x_x = torch.transpose(x, 1, 2)  # x_x = torch.transpose(x, 1, 2)
        x, (h_n, h_c) = self.BiLSTM(x_x)  # torch.Size([100, 75, 640])

        x = torch.flatten(x, 1)
        x = self.Linear1(x)
        x = F.relu(x)

        x = self.Linear2(x)

        #x = torch.sigmoid(x)
        return x



# similar as vgg16, kernal size remains, channels increase
class CnnActivation2(nn.Module):
    def __init__(self, n_class):

        super(CnnActivation2, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(960, n_class)

    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 320, 993]
        x = self.Conv1(input)
        x = F.relu(x)
       # print(x.shape)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 320, 993]
        # Output Tensor Shape: [batch_size, 320, 248]
        x = self.Maxpool(x)
        x = self.Drop1(x)
      #  print(x.shape)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 320, 248]
        # Output Tensor Shape: [batch_size, 480, 241]
        x = self.Conv2(x)
        x = F.relu(x)
       # print(x.shape)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 480, 241]
        # Output Tensor Shape: [batch_size, 480, 60]
        x = self.Maxpool(x)
        x = self.Drop1(x)
      #  print(x.shape)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 480, 53]
        # Output Tensor Shape: [batch_size, 960, 53]
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
       # print(x.shape)

        # Input Tensor Shape: [batch_size, 960, 53]
        # Output Tensor Shape: [batch_size, 960, 1]
        x = torch.mean(x, 2)

        #print(x.shape)

        x = self.Linear1(x)

        return x
