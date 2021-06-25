from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, n_class):
        super(BasicBlock, self).__init__()
        self.Conv0 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=9, padding=4)
        self.Conv1 = nn.Conv1d(in_channels=320, out_channels=320, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(320)
        # self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=9, padding=4)
        # self.bn2 = nn.BatchNorm1d(480)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(480)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=480, kernel_size=5, padding=2)
        #self.bn4 = nn.BatchNorm1d(1024)
        self.Maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.Maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        #self.Drop1 = nn.Dropout(p=0.2)
        #self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(31*480, 925)
        self.Linear2 = nn.Linear(925, n_class)

    def forward(self, input):

        # Convolution Layer 1
        # Input Tensor Shape: [batch_size, 4, 1000]
        # Output Tensor Shape: [batch_size, 320, 1000]
        x = self.Conv0(input)
        x = self.bn1(x)
        x = F.relu(x)

        identity = x

        # Convolution Layer 2
        # Input Tensor Shape: [batch_size, 320, 1000]
        # Output Tensor Shape: [batch_size, 320, 1000]
        x = self.Conv1(x)
        x = self.bn1(x)

        x += identity

        x = F.relu(x)

        # Pooling Layer
        # Input Tensor Shape: [batch_size, 320, 1000]
        # Output Tensor Shape: [batch_size, 320, 125]
        x = self.Maxpool1(x)


        # Convolution Layer 3
        # Input Tensor Shape: [batch_size, 480, 125]
        # Output Tensor Shape: [batch_size, 480, 125]
        x = self.Conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        identity = x

        # Convolution Layer 4
        # Input Tensor Shape: [batch_size, 480, 125]
        # Output Tensor Shape: [batch_size, 480, 125]
        x = self.Conv3(x)
        x = self.bn2(x)

        x += identity

        x = F.relu(x)


        # Pooling Layer 3
        # Input Tensor Shape: [batch_size, 480, 125]
        # Output Tensor Shape: [batch_size, 480, 31]
        x = self.Maxpool2(x)

        x = x.view(-1, 31*480)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x
