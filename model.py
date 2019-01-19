import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

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
        self.act8 = nn.Softmax(7)




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

class ModelGithub(nn.Module):

    def __init__(self):
        super(ModelGithub, self).__init__()

        hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8
        )
        self._classifier = nn.Sequential(
            hidden9,
            hidden10
        )
        self._digit_length = nn.Sequential(nn.Linear(3072, 7))
        self._digit1 = nn.Sequential(nn.Linear(3072, 11))
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 192 * 7 * 7)
        x = self._classifier(x)

        length_logits, digits_logits = self._digit_length(x), [self._digit1(x),
                                                               self._digit2(x),
                                                               self._digit3(x),
                                                               self._digit4(x),
                                                               self._digit5(x)]

        return length_logits