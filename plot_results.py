"""
Plot the training and validation losses.

The losses are read from a .TXT file. The file should be format as

    epoch <epoch id> loss <loss value>

To run this script:

    python plot_results.py -dir results/senet -m senet

    python plot_results.py -dir results/resnet -m resnet

    python plot_results.py -dir results/goodfellow -m goodfellow

"""
import re
import argparse
import matplotlib.pyplot as plt

def argparser():
    """
    Command line argument parser
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
    Read a text file
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
    Read a text file
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
    plt.plot(train_acc, color='black')
    plt.plot(valid_acc, color='red')
    plt.legend(['train accuracy', 'validation accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./figures/acc_{}.png'.format(model_name))
    plt.close()

def plot_loss(train_loss, valid_loss, model_name):
    plt.plot(train_loss, color='black')
    plt.plot(valid_loss, color='red')
    plt.legend(['train loss', 'validation loss'])
    plt.xlabel('epoch')
    plt.ylabel('-log p(y|x)')
    plt.savefig('./figures/loss_{}.png'.format(model_name))
    plt.close()


def main(args):
    train_acc = read_accuracy_file('{}/{}'.format(args.results_dir, args.train_acc_file))
    valid_acc = read_accuracy_file('{}/{}'.format(args.results_dir, args.valid_acc_file))

    train_loss = read_loss_file('{}/{}'.format(args.results_dir, args.train_loss_file))
    valid_loss = read_loss_file('{}/{}'.format(args.results_dir, args.valid_loss_file))

    plot_accuracy(train_acc, valid_acc, args.model)

    plot_loss(train_loss, valid_loss, args.model)


    

if __name__ == '__main__':
    main(argparser())