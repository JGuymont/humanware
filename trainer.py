import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from models.large_cnn import LargeCNN
from models.medium_cnn import MediumCNN
from models.small_cnn import SmallCNN
import pandas as pd

class Trainer:
    def __init__(self, conf):
        self.epochs = conf.getint("n_epochs")
        self.epoch = conf.getint("epoch_start")
        self.lr = conf.getfloat("learning_rate")
        self.batch_size = conf.getint("batch_size")
        self.criterion = nn.CrossEntropyLoss()
        self.model = eval(conf.get('model'))(conf).to(conf.get('device'))
        self.device = conf.get('device')
        if conf.get("optim") == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                       momentum=conf.getfloat("momentum"),
                                       weight_decay=conf.getfloat("weight_decay"))
        else:
            # TODO: add Adam
            raise ValueError('Only SGD is supported')

    def train_model(self, trainloader, devloader):
        self.train_size = sum([x.shape[0] for x, _ in trainloader])
        self.valid_size = sum([x.shape[0] for x, _ in devloader])
        for _ in range(self.epochs):
            self.run_epoch(trainloader, devloader)

    def run_epoch(self, trainloader, devloader):
        self.accuracies_train = []
        for x_batch, target_batch in trainloader:      
            self.train_on_batch(x_batch.to(self.device), target_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_train) / self.train_size
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()
        print(" [*] epoch {} train accuracy {}".format(self.epoch, total_accuracy_for_epoch))

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
        correct = np.sum(pred == real)
        self.accuracies_train.append(correct)

        txt_file = open("results/train_loss.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, loss.item()))
        txt_file.close()

    def validation(self, devloader):
        self.accuracies_val = []
        with torch.no_grad():
            for x_batch, y_batch in devloader:
                self.validation_batch(x_batch.to(self.device), y_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_val) / self.valid_size
        txt_file = open("results/val_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()
        print(" [*] epoch {} valid accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))

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