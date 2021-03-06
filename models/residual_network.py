"""
Deep Residual Network implementation in Pytorch
"""
import json

from torch import nn


def conv3x3(in_channels, out_channels, stride=1):
    """
    3x3 convolution

    :param in_channels: the number of input channels
    :param out_channels: the number of output channels
    :param stride: the size of the stride
    :return: the output of the convolution
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Class representing a residual block
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Instantiate a residual block.

        :param in_channels: the number of input channels
        :param out_channels: the number of output channels
        :param stride: the size of the stride
        :param downsample: object that will downsample the input
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """
        Function to apply the forward pass of the residual block.

        :param x: the input
        :return: the output
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    Residual Network
    """
    def __init__(self, config, block=ResidualBlock):
        """
        Instantiate a ResNet

        :param config: configuration the ResNet (size of the output layer)
        :param block: the type of residual block
        """
        super(ResNet, self).__init__()
        layers = json.loads(config.get("layers"))
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, config.getint("num_classes"))

    def make_layer(self, block, out_channels, blocks, stride=1):
        """
        Function to create one layer of the ResNet.

        :param block: the type of residual block
        :param out_channels: the number of out channels
        :param blocks: the number of blocks in the layer
        :param stride: the size of the stride
        :return: a sequential containing all the sub-layers
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride,
                            downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Function to apply the forward pass of the ResNet

        :param x: the input
        :return: the output
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
