import pickle
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

METADATA_PATH = './data/SVHN/train_metadata.pkl'

def load_pickle(path):
    with open(path, 'rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def tensor_to_image(tensor):
    img = transforms.ToPILImage(mode='RGB')(tensor)
    img.show()

def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist

def bar_plot(data, xtitle, title):
    label = list(set(data))
    height = count_elements(data)
    height = [height[i] for i in label]
    plt.bar(label, height=height, width=0.8)
    plt.ylabel('frequency')
    plt.xlabel(xtitle)
    plt.xticks(label)
    plt.savefig('./figures/{}.png'.format(title))
    plt.close()

def main(metadata_path):
    metadata = load_pickle(metadata_path)
    labels = [example['metadata']['label'] for example in metadata.values()]
    lenghts = [len(label) for label in labels]
    digits = [int(digit) for label in labels for digit in label]

    bar_plot(lenghts, xtitle='lenght', title='lenghts')
    bar_plot(digits, xtitle='digit', title='digits')

    

if __name__ == '__main__':
    main(METADATA_PATH)
