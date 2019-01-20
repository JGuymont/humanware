import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from models.cnn import ConvNet
from models.model_paper import ModelPaper
import pandas as pd

class Trainer:
    def __init__(self, args):
        self.epochs = args.n_epochs
        self.epoch = 0
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.model = eval(args.model)(args).to(args.device)
        self.device = args.device
        if args.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        else:
            # TODO: add Adam
            raise ValueError('Only SGD is supported')

    def train_model(self, trainloader, devloader):
        print(' [*] Starting training')
        for _ in tqdm(range(self.epochs)):
            self.run_epoch(trainloader, devloader)

    def run_epoch(self, trainloader, devloader):
        self.accuracies_train = []
        for x_batch, target_batch in tqdm(trainloader):      
            self.train_on_batch(x_batch.to(self.device), target_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_train) / len(trainloader)
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()

        self.validation(devloader)
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

    def validation(self, devloader):
        print(' [*] Computing validation accuracy')
        self.accuracies_val = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(devloader):
                self.validation_batch(x_batch.to(self.device), y_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_val) / len(devloader)
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

    def test_run(self, testloader):
        self.accuracies_test = []
        with torch.no_grad():
            for x_batch, y_batch in tqdm(testloader):
                self.make_predictions(x_batch.to(self.device), y_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_test) / len(testloader)
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