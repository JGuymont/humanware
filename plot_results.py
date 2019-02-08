"""
Plot the training and validation losses and accuracies.

To run this script:
```
python plot_results.py -dir results/senet -m senet
python plot_results.py -dir results/resnet -m resnet
python plot_results.py -dir results/goodfellow -m goodfellow
```
"""
import re
import argparse
import matplotlib.pyplot as plt

def argparser():
    """
    Configure the command-line arguments parser

    :return: the arguments parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', '-dir', type=str)
    
    parser.add_argument('--train_acc_file', type=str, default='train_accuracies.txt')
    parser.add_argument('--valid_acc_file', type=str, default='valid_accuracies.txt')
    
    parser.add_argument('--train_loss_file', type=str, default='train_losses.txt')
    parser.add_argument('--valid_loss_file', type=str, default='valid_losses.txt')
    
    parser.add_argument('--out_dir', type=str, default='./figures')
    parser.add_argument('--model', '-m', type=str)
    return parser.parse_args()

def read_accuracy_file(filename):
    """
    Read a text file containing the accuracies.

    :param filename: the path to the .txt file containing the accuracies.
        This file is generated during training and the line are formated 
        as `epoch <epoch> accuracy <accuracy>`

    :return: list of accuracies
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    accuracies = []
    for line in content:
        acc = re.search('accuracy\s(.+?)$', line)
        accuracies.append(float(acc.group(1)))
    return accuracies

def read_loss_file(filename):
    """
    Read a text file containing the losses.

    :param filename: the path to the .txt file containing the losses.
        This file is generated during training and the line are formated 
        as `epoch <epoch> loss <loss>`

    :return: list of losses
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    losses = []
    for line in content:
        loss = re.search('loss\s(.+?)$', line)
        losses.append(float(loss.group(1)))
    return losses

def plot_accuracy(train_acc, valid_acc, model_name):
    """
    Plot the accuracies of the training and validation set
    accross epochs and save the plot under './figures/acc_<model_name>.png'

    :param train_acc: list of training accuracies
    :param valid_acc: list of validation accuracies
    :param model_name: name of the model (string)
    """
    plt.plot(train_acc, color='black')
    plt.plot(valid_acc, color='red')
    plt.legend(['train accuracy', 'validation accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./figures/acc_{}.png'.format(model_name))
    plt.close()

def plot_loss(train_loss, valid_loss, model_name):
    """
    Plot the losses of the training and validation set
    accross epochs and save the plot under './figures/loss_<model_name>.png'

    :param train_acc: list of training losses
    :param valid_acc: list of validation losses
    :param model_name: name of the model (string)
    """
    plt.plot(train_loss, color='black')
    plt.plot(valid_loss, color='red')
    plt.legend(['train loss', 'validation loss'])
    plt.xlabel('epoch')
    plt.ylabel('-log p(y|x)')
    plt.savefig('./figures/loss_{}.png'.format(model_name))
    plt.close()


if __name__ == '__main__':
    args = argparser()

    train_acc = read_accuracy_file('{}/{}'.format(args.results_dir, args.train_acc_file))
    valid_acc = read_accuracy_file('{}/{}'.format(args.results_dir, args.valid_acc_file))

    train_loss = read_loss_file('{}/{}'.format(args.results_dir, args.train_loss_file))
    valid_loss = read_loss_file('{}/{}'.format(args.results_dir, args.valid_loss_file))

    plot_accuracy(train_acc, valid_acc, args.model)

    plot_loss(train_loss, valid_loss, args.model)