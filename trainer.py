import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from models.large_cnn import LargeCNN
from models.medium_cnn import MediumCNN
from models.small_cnn import SmallCNN
from models.residual_network import ResNet
import pandas as pd
import os
import shutil

class Trainer:
    def __init__(self, conf):
        model_conf = conf["model"]
        self.epochs = model_conf.getint("n_epochs")
        self.epoch = model_conf.getint("epoch_start")
        self.lr = model_conf.getfloat("learning_rate")
        self.batch_size = model_conf.getint("batch_size")
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(model_conf.get('device'))
        self.model = eval(model_conf.get('name'))(model_conf).to(self.device)
        if model_conf.get("optim") == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                                       momentum=model_conf.getfloat("momentum"),
                                       weight_decay=model_conf.getfloat("weight_decay"))
        else:
            # TODO: add Adam
            raise ValueError('Only SGD is supported')
        self.checkpoints_path = model_conf.get("checkpoints_path")
        self.best_accuracy = 0

    def train_model(self, trainloader, devloader):
        self.train_size = sum([x.shape[0] for x, _ in trainloader])
        self.valid_size = sum([x.shape[0] for x, _ in devloader])
        for _ in tqdm(range(self.epochs)):
            self.run_epoch(trainloader, devloader)

    def run_epoch(self, trainloader, devloader):
        self.accuracies_train = []
        for x_batch, target_batch in tqdm(trainloader):      
            self.train_on_batch(x_batch.to(self.device), target_batch.to(self.device))

        total_accuracy_for_epoch = np.sum(self.accuracies_train) / self.train_size
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()
        print(" [*] epoch {} train accuracy {}".format(self.epoch, total_accuracy_for_epoch))

        total_accuracy_for_epoch = self.validation(devloader)
        self.save_checkpoint(total_accuracy_for_epoch)
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
        return total_accuracy_for_epoch

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

    def save_checkpoint(self, accuracy):
        state_dict = {'epoch': self.epoch + 1,
                       'state_dict': self.model.state_dict(),
                       'optim_dict' : self.optimizer.state_dict()}
        torch.save(state_dict, os.path.join(self.checkpoints_path, "last_{:+.2f}.pth".format(accuracy)))
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            shutil.copyfile(os.path.join(self.checkpoints_path, "best_{:+.2f}.pth".format(accuracy)))