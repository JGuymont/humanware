import pickle
import pandas as pd
from utils import TrainSVHNDataset
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

class CNN:
    def __init__(self, lr=0.001, batch_size=32, total_epochs=50):
        self.epochs=total_epochs
        self.lr =lr
        self.batch_size = batch_size

        self.train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.train_dataset=TrainSVHNDataset(transform=transforms, root=None)
        print(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True)
        self.criterion = nn.CrossEntropyLoss


    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            self.train_batch()

    def train_batch(self):
        for x_batch, target_batch in tqdm(self.train_loader):
            self.train_on_batch(x_batch.to(device), y_batch.to(device))

    def train_on_batch(self, x, y):
        pass

if __name__ == '__main__':



    trainer = CNN(lr=0.001, batch_size=1)
    trainer.train_model()