import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

class MediumCNN(nn.Module):
    def __init__(self):
        super(MediumCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=3, bias=False)
        self.conv4 = nn.Conv2d(128, 192, kernel_size=5, stride=1, padding=3, bias=False)

        self.fc1 = nn.Linear(4*192, 2)
        self.fc2 = nn.Linear(3072, 768)
        self.fc3 = nn.Linear(768, 192)
        self.fc4 = nn.Linear(192, 7)

        self.act1 = nn.PReLU(48)
        self.act2 = nn.PReLU(64)
        self.act3 = nn.PReLU(128)
        self.act4 = nn.PReLU(192)
        self.act5 = nn.PReLU(3072)
        self.act6 = nn.PReLU(768)
        self.act7 = nn.PReLU(192)
        self.act8 = nn.Softmax(5)

    def forward(self, x):
        x=self.conv1(x)
        x = self.act1(x)
        x=F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x =self.act3(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = self.act4(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, x.shape[0])
        x=self.fc1(x)
        x = self.act5(x)

        x = self.fc2(x)
        x = self.act6(x)

        x = self.fc3(x)
        x = self.act7(x)

        x = self.fc4(x)
        x = self.act8(x)
        return x