import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from models.large_cnn import LargeCNN
from models.medium_cnn import MediumCNN
from models.small_cnn import SmallCNN
from models.senet import senet
import pandas as pd
import os
import shutil


class Trainer:
    def __init__(self, conf):
        self.print_freq = conf.getint("log", "print_freq")
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
        elif model_conf.get("optim") == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if model_conf.get("checkpoint") is not None:
            self.load_checkpoint(model_conf.get("checkpoint"))

        self.checkpoints_path = model_conf.get("checkpoints_path")
        self.best_accuracy = 0
        self.last_accuracy = 0

    def train_model(self, trainloader, devloader):
        print(len(trainloader.dataset))
        self.model.train()
        self.train_size = len(trainloader.dataset)
        self.valid_size = len(devloader.dataset)
        for _ in range(self.epochs):
            self.run_epoch(trainloader, devloader)

    def run_epoch(self, trainloader, devloader):
        self.accuracies_train = 0
        for i, (x_batch, target_batch) in enumerate(trainloader):
            self.train_on_batch(i, x_batch.to(self.device), target_batch.to(self.device))

        total_accuracy_for_epoch = self.accuracies_train / float((i + 1) * self.batch_size)
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()
        print(" [*] epoch {}/{} train accuracy {}".format(self.epoch, self.epochs, total_accuracy_for_epoch))

        total_accuracy_for_epoch = self.validation(devloader)
        self.save_checkpoint(total_accuracy_for_epoch)
        self.epoch += 1

    def train_on_batch(self, iteration, x, y):

        output = self.model(x)
        loss = self.criterion(output, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        real = y.detach().cpu().numpy()
        correct = np.sum(pred == real)
        self.accuracies_train += correct

        txt_file = open("results/train_loss.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, loss.item()))
        txt_file.close()

        if iteration % self.print_freq == 0:
            print("iteration {it}/{total} train accuracy {acc:.4f}({mean_acc:.4f}) loss {loss}".format(it=iteration,
                                                                                                       total=self.train_size // self.batch_size,
                                                                                                       acc=correct * 100 / self.batch_size,
                                                                                                       mean_acc=self.accuracies_train * 100 / float(
                                                                                                           (
                                                                                                           iteration + 1) * self.batch_size),
                                                                                                       loss=loss.item()))

    def validation(self, devloader):
        self.accuracies_val = 0
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(devloader):
                self.validation_batch(i, x_batch.to(self.device), y_batch.to(self.device))

        total_accuracy_for_epoch = self.accuracies_val / float((i + 1) * 100)
        txt_file = open("results/val_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        txt_file.close()
        print(" [*] epoch {} valid accuracy {} \n".format(self.epoch, total_accuracy_for_epoch))
        return total_accuracy_for_epoch

    def validation_batch(self, iteration, x, y):
        self.model.eval()
        output = self.model(x)
        loss = self.criterion(output, y)

        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        real = y.detach().cpu().numpy()
        accuracy = np.sum(pred == real)
        self.accuracies_val += accuracy

        txt_file = open("results/loss_val.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, loss.item()))
        txt_file.close()

        if iteration % self.print_freq == 0:
            print("iteration {it}/{total} valid accuracy {acc:.4f}({mean_acc:.4f}) loss {loss}".format(it=iteration,
                                                                                                       total=self.valid_size // 100,
                                                                                                       acc=accuracy,
                                                                                                       mean_acc=self.accuracies_val / float(
                                                                                                           iteration + 1),
                                                                                                       loss=loss.item()))

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
                      'optim_dict': self.optimizer.state_dict()}
        torch.save(state_dict, os.path.join(self.checkpoints_path, "last.pth"))

        if self.last_accuracy > 0:
            os.remove(os.path.join(self.checkpoints_path, "last.pth"))
            self.last_accuracy = accuracy

        if accuracy > self.best_accuracy:
            if self.best_accuracy > 0:
                os.remove(os.path.join(self.checkpoints_path, "best_{acc:.4f}.pth".format(acc=self.best_accuracy)))
            self.best_accuracy = accuracy
            torch.save(state_dict, os.path.join(self.checkpoints_path, "best_{acc:.4f}.pth".format(acc=accuracy)))

    def load_checkpoint(self, checkpoint_path, continue_from_epoch=True):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optim_dict'])

        if continue_from_epoch:
            self.epoch = state['epoch']