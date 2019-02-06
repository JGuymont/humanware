"""
Implementation of the CNN describe in Goodfellow et al, 2013
"""
import torch.nn as nn

# from https://github.com/pytorch/pytorch/issues/805
# not used - but left here for reference on how to
# make Maxout units like in Goodfellow et al, 2013
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert (x.shape[-1] % self._pool_size == 0,
            'Wrong input last dim size ({}) for Maxout({})'
                .format(x.shape[-1], self._pool_size))
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size,
                      self._pool_size).max(2)
        return m

class CNNpaper(nn.Module):
    """
    Class representing the model from Goodfellow et al, 2013
    """
    def __init__(self, conf):
        """
        Instantiate the model's layers.

        :param conf: model configuration
        """
        super(CNNpaper, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 5, padding=2),
            nn.BatchNorm2d(48),
            # Relu (min = 0) followed by maxpool2d is similar to maxout
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 160, 5, padding=2),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(160, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 192, 5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1),
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9408, 3072),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            self.fc1,
            self.fc2
        )
        self.classify = nn.Sequential(nn.Linear(3072,
                                                conf.getint("num_classes")))

    def forward(self, x):
        """
        Function to apply the forward pass of the residual block.

        :param x: the input
        :return: the output
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc(x)

        length = self.classify(x)

        return length
