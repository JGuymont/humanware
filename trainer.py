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
import json
import time

class Trainer:
    def __init__(self, conf):
        model_conf = conf["model"]
        self.epochs = model_conf.getint("n_epochs")
        self.epoch = model_conf.getint("epoch_start")
        self.batch_size = model_conf.getint("batch_size")
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(model_conf.get('device'))
        self.model = eval(model_conf.get('name'))(model_conf).to(self.device)
        if model_conf.get("optim") == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=model_conf.getfloat("learning_rate"),
                momentum=model_conf.getfloat("momentum"),
                weight_decay=model_conf.getfloat("weight_decay"))
        elif model_conf.get("optim") == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=model_conf.getfloat("learning_rate"),
                betas=json.loads(conf["model"].get("betas")))
        else:
            raise ValueError('Only SGD is supported')
        self.checkpoints_path = model_conf.get("checkpoints_path")
        self.best_accuracy = 0
        self.train_size = None
        self.valid_size = None

    def train_model(self, trainloader, devloader):
        """
        Find the optimal parameters according to self.criterion
        using SGD
        """
        start_time = time.time()

        for _ in range(self.epochs):
            self.model.train()
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            train_acc, train_loss = self.evaluate(trainloader)
            valid_acc, valid_loss = self.evaluate(devloader)

            self.save_checkpoint(valid_acc)

            self._log_epoch(train_acc, train_loss, valid_acc, valid_loss)

            print(' [*] Epoch: {:.0f} | Loss: {:.3f} | Train acc: {:.2f} | Dev acc: {:.2f} | time: {} sec.'.format(
                self.epoch+1, train_loss, train_acc, valid_acc, round(time.time() - start_time)))

            self.epoch += 1

    def evaluate(self, dataloader):
        self.model.eval()
        losses = []
        total, correct = 0., 0.
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
        txt_file = open("results/train_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, train_acc))
        txt_file.close()

        txt_file = open("results/train_loss.txt", "a")
        txt_file.write("epoch {} loss {} \n".format(self.epoch, train_loss))
        txt_file.close()

        txt_file = open("results/val_accuracies.txt", "a")
        txt_file.write("epoch {} accuracy {} \n".format(self.epoch, valid_acc))
        txt_file.close()

        txt_file = open("results/loss_val.txt", "a")
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
        torch.save(state_dict, os.path.join(self.checkpoints_path, "last_{:+.2f}.pth".format(accuracy)))
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            shutil.copyfile(
                src=os.path.join(self.checkpoints_path, "last_{:+.2f}.pth".format(accuracy)),
                dst=os.path.join(self.checkpoints_path, "best_{:+.2f}.pth".format(accuracy))
            )