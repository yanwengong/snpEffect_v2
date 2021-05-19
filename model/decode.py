import torch
from torch import nn
import torch.nn.functional as F


class Decode(nn.Module):
    def __init__(self, n_class):

        super(Decode, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=8)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Conv3 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv4 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=4)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Conv5 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=4)
        self.Conv6 = nn.Conv1d(in_channels=960, out_channels=960, kernel_size=4)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)

        self.Linear1 = nn.Linear(12480, 1024)
        self.Drop2 = nn.Dropout(p=0.4)

        self.Linear2 = nn.Linear(1024, n_class)

    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 320, 993]
        x = self.Conv1(input)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 320, 993]
        # Output Tensor Shape: [batch_size, 320, 986]
        x = self.Conv2(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 320, 986]
        # Output Tensor Shape: [batch_size, 320, 246]
        x = self.Maxpool1(x)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 320, 246]
        # Output Tensor Shape: [batch_size, 480, 239]
        x = self.Conv3(x)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 480, 239]
        # Output Tensor Shape: [batch_size, 480, 236]
        x = self.Conv4(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 480, 236]
        # Output Tensor Shape: [batch_size, 480, 59]
        x = self.Maxpool2(x)

        # Convolution Layer 5
        # Input Tensor Shape: [batch_size, 480, 59]
        # Output Tensor Shape: [batch_size, 960, 56]
        x = self.Conv5(x)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 6
        # Input Tensor Shape: [batch_size, 960, 56]
        # Output Tensor Shape: [batch_size, 960, 53]
        x = self.Conv6(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 960, 53]
        # Output Tensor Shape: [batch_size, 960, 13]
        x = self.Maxpool3(x)

        x = self.Drop1(x)

        # print(x.shape)

        ## Flatten
        # Input Tensor Shape: [batch_size, 960, 13]
        # Output Tensor Shape: [batch_size, 12480]
        x = torch.flatten(x, 1)
        # Linear Layer 1
        # Input Tensor Shape: [batch_size, 12480]
        # Output Tensor Shape: [batch_size, 1024]
        x = self.Linear1(x)

        x = self.Drop2(x)

        # Linear Layer 2
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, n_class]
        x = torch.flatten(x, 1)
        # Linear Layer 1
        x = self.Linear2(x)

        return x



class Decode2(nn.Module):
    def __init__(self, n_class):

        super(Decode2, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=128, out_channels=360, kernel_size=8)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Conv3 = nn.Conv1d(in_channels=360, out_channels=360, kernel_size=8)
        self.Conv4 = nn.Conv1d(in_channels=360, out_channels=720, kernel_size=4)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Conv5 = nn.Conv1d(in_channels=720, out_channels=960, kernel_size=4)
        self.Conv6 = nn.Conv1d(in_channels=960, out_channels=960, kernel_size=4)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)

        self.Linear1 = nn.Linear(12480, 1024)
        self.Drop2 = nn.Dropout(p=0.4)

        self.Linear2 = nn.Linear(1024, n_class)

    def forward(self, input):
        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 128, 993]
        x = self.Conv1(input)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 128, 993]
        # Output Tensor Shape: [batch_size, 360, 986]
        x = self.Conv2(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 1
        # Input Tensor Shape: [batch_size, 360, 986]
        # Output Tensor Shape: [batch_size, 360, 246]
        x = self.Maxpool1(x)

        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 360, 246]
        # Output Tensor Shape: [batch_size, 360, 239]
        x = self.Conv3(x)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 360, 239]
        # Output Tensor Shape: [batch_size, 720, 236]
        x = self.Conv4(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 2
        # Input Tensor Shape: [batch_size, 720, 236]
        # Output Tensor Shape: [batch_size, 720, 59]
        x = self.Maxpool2(x)
        x = self.Drop1(x)

        # Convolution Layer 5
        # Input Tensor Shape: [batch_size, 720, 59]
        # Output Tensor Shape: [batch_size, 960, 56]
        x = self.Conv5(x)
        x = F.relu(x)
        # print(x.shape)

        # Convolution Layer 6
        # Input Tensor Shape: [batch_size, 960, 56]
        # Output Tensor Shape: [batch_size, 960, 53]
        x = self.Conv6(x)
        x = F.relu(x)
        # print(x.shape)

        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 960, 53]
        # Output Tensor Shape: [batch_size, 960, 13]
        x = self.Maxpool3(x)

        x = self.Drop1(x)

        # print(x.shape)

        ## Flatten
        # Input Tensor Shape: [batch_size, 960, 13]
        # Output Tensor Shape: [batch_size, 12480]
        x = torch.flatten(x, 1)
        # Linear Layer 1
        # Input Tensor Shape: [batch_size, 12480]
        # Output Tensor Shape: [batch_size, 1024]
        x = self.Linear1(x)

        x = self.Drop2(x)

        # Linear Layer 2
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, n_class]
        x = self.Linear2(x)

        return x