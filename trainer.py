import torch.nn as nn
from tqdm import tqdm
import torch
import torch.optim as optim
import numpy as np
from models.large_cnn import CNNpaper
from models.medium_cnn import MediumCNN
from models.small_cnn import SmallCNN
from models.residual_network import ResNet
from models.senet import senet
import pandas as pd
import os
import shutil
import json
import time

class Trainer:
    """
    Optimize the parameters of a model by minimizing
    an objective functions.

    Args
        config (.INI file): Configuration file containing the
            hyperparameters of the model.
    """
    def __init__(self, conf):
        self.model_conf = conf["model"]
        self.epochs = self.model_conf.getint("n_epochs")
        self.epoch = self.model_conf.getint("epoch_start")
        self.batch_size = self.model_conf.getint("batch_size")
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(self.model_conf.get('device'))
        self.model = eval(self.model_conf.get('name'))(self.model_conf).to(self.device)
        # self.model = nn.DataParallel(eval(self.model_conf.get('name'))(self.model_conf).to(self.device))
        if self.model_conf.get("optim") == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.model_conf.getfloat("learning_rate"),
                momentum=self.model_conf.getfloat("momentum"),
                weight_decay=self.model_conf.getfloat("weight_decay"))
        elif self.model_conf.get("optim") == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.model_conf.getfloat("learning_rate"),
                betas=json.loads(self.model_conf.get("betas")))
        else:
            raise ValueError('Only SGD is supported')

        if self.model_conf.get("checkpoint") is not None:
            self.load_checkpoint(self.model_conf.get("checkpoint"))

        self.checkpoints_path = conf.get("paths", "checkpoints")
        self.results_path = conf.get("paths", "results")
        self.best_accuracy = 0
        self.train_size = None
        self.valid_size = None
        self.iteration_print_freq = conf.getint("log", "iteration_print_freq")

    def train_model(self, trainloader, devloader):
        """
        Find the optimal parameters according to self.criterion
        using SGD
        """
        start_time = time.time()

        for self.epoch in range(self.epoch, self.epochs):
            self.model.train()
            accuracy_train = 0
            loss_train = 0
            for batch_index, (x_batch, y_batch) in enumerate(trainloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_train += loss.item()

                pred = np.argmax(output.detach().cpu().numpy(), axis=1)
                real = y_batch.detach().cpu().numpy()
                correct = np.sum(pred == real)
                accuracy_train += correct

                if batch_index % self.iteration_print_freq == 0:
                    print("Iteration: {:.0f}/{} | Train Loss: {:.4f} ({:.4f}) | Train Accuracy: {:.2f} ({:.4f})"
                          .format(batch_index, len(trainloader), loss, loss_train / (batch_index + 1),
                                  correct * 100 / self.batch_size,
                                  accuracy_train * 100 / ((batch_index + 1) * self.batch_size)))

            train_acc = accuracy_train * 100 / ((batch_index + 1) * self.batch_size)
            train_loss = loss_train / (batch_index + 1)
            # train_acc, train_loss = self.evaluate(trainloader)

            valid_acc, valid_loss = self.evaluate(devloader)

            # self.save_checkpoint(valid_acc)

            self._log_epoch(train_acc, train_loss, valid_acc, valid_loss)

            print(' [*] Epoch: {:.0f} | Loss: {:.3f} | Train acc: {:.2f} | Dev acc: {:.2f} | time: {} sec.'.format(
                self.epoch+1, train_loss, train_acc, valid_acc, round(time.time() - start_time)))

    def evaluate(self, dataloader):
        self.model.eval()
        losses = []
        total, correct = 0., 0.
        with torch.no_grad():
            for (inputs, targets) in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().item()

        accuracy = 100. * correct / total
        return round(accuracy, 4), np.mean(losses)

    def _log_epoch(self, train_acc, train_loss, valid_acc, valid_loss):
        txt_file = open(os.path.join(self.results_path, "train_accuracies.txt"), "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, train_acc))
        txt_file.close()

        txt_file = open(os.path.join(self.results_path, "train_loss.txt"), "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, train_loss))
        txt_file.close()

        txt_file = open(os.path.join(self.results_path, "val_accuracies.txt"), "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, valid_acc))
        txt_file.close()

        txt_file = open(os.path.join(self.results_path, "loss_val.txt"), "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, valid_loss))
        txt_file.close()

    def make_predictions(self, x, y):
        self.model.eval()
        output = self.model(x)
        pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        return pred

    def save_checkpoint(self, accuracy):
        state_dict = {
            'epoch': self.epoch + 1,
            'state_dict': self.model.state_dict(),
            'optim_dict' : self.optimizer.state_dict()
        }
        torch.save(state_dict, os.path.join(self.checkpoints_path, "last.pth".format(accuracy)))
        if accuracy > self.best_accuracy:
            if self.best_accuracy > 0:
                os.remove(os.path.join(self.checkpoints_path, "best_{acc:.4f}.pth".format(acc=self.best_accuracy)))
            self.best_accuracy = accuracy
            torch.save(state_dict, os.path.join(self.checkpoints_path, "best_{acc:.4f}.pth".format(acc=accuracy)))
            self.best_accuracy = accuracy

    def load_checkpoint(self, checkpoint_path, continue_from_epoch=True):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optim_dict'])

        if continue_from_epoch:
            self.epoch = state['epoch']