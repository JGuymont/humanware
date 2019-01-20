import pickle
import pandas as pd
from utils import TrainSVHNDataset, ValSVHNDataset, SyntheticTestSVHNDataset
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.optim as optim
from model import SimpleConvNet, ModelPaper
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


class CNN:
    def __init__(self, batch_size, model, lr=0.001, total_epochs=50):
        self.epochs = total_epochs
        self.epoch = 0
        self.lr = lr
        self.batch_size = batch_size

        self.train_transforms = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.RandomApply([
                transforms.RandomAffine(degrees=30, shear=20),
                transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=.5),
                transforms.RandomRotation(20),
            ], p=0.5),
            transforms.RandomResizedCrop(54),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.test_transforms = transforms.Compose([
            transforms.CenterCrop(54),
            transforms.ToTensor(),
        ])

        self.train_dataset = TrainSVHNDataset(transform=self.train_transforms, root=None)
        self.val_dataset = ValSVHNDataset(transform=self.test_transforms, root=None)
        self.test_dataset = SyntheticTestSVHNDataset(transform=self.test_transforms, root=None)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=2, pin_memory=False)

        self.criterion = nn.CrossEntropyLoss()
        self.model = eval(model)().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            self.run_epoch()

    def run_epoch(self):
        self.accuracies_train = []
        for x_batch, target_batch in tqdm(self.train_loader):
            # print(type(x_batch), type(target_batch), target_batch)
            self.train_on_batch(x_batch.to(device), target_batch.to(device))

        total_accuracy_for_epoch = np.sum(self.accuracies_train) / pd.read_pickle('train_filenames.pkl').shape[0]
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()

        self.validation()
        self.test_run()
        self.epoch += 1

    def train_on_batch(self, x, y):
        self.model.train()
        output = self.model(x)
        loss = self.criterion(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        real = y.detach().cpu().numpy()
        accuracy = np.sum(pred == real)
        self.accuracies_train.append(accuracy)

        txt_file = open("results/train_loss.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, loss.item()))
        txt_file.close()

    def validation(self):
        self.accuracies_val = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.val_loader):
                self.validation_batch(x_batch.to(device), y_batch.to(device))

        total_accuracy_for_epoch = np.sum(self.accuracies_val) / pd.read_pickle('val_filenames.pkl').shape[0]
        txt_file = open("results/val_accuracies.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()

    def validation_batch(self, x, y):
        self.model.eval()
        output = self.model(x)
        loss = self.criterion(output, y)

        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        real = y.detach().cpu().numpy()
        accuracy = np.sum(pred == real)
        self.accuracies_val.append(accuracy)

        txt_file = open("results/loss_val.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, loss.item()))
        txt_file.close()

    def test_run(self):
        self.accuracies_test = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.test_loader):
                self.make_predictions(x_batch.to(device), y_batch.to(device))

        total_accuracy_for_epoch = np.sum(self.accuracies_test) / pd.read_pickle('test_filenames.pkl').shape[0]
        txt_file = open("results/test_accuracies.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()

    def make_predictions(self, x, y):
        self.model.eval()
        output = self.model(x)
        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        real = y.detach().cpu().numpy()
        accuracy = np.sum(pred == real)
        self.accuracies_test.append(accuracy)


if __name__ == '__main__':
    # result folder for
    if not os.path.isdir('results'):
        os.mkdir('results')

    trainer = CNN(32, model='ModelPaper', lr=0.001, total_epochs=150)
    trainer.train_model()
