import pickle
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from data import SVHNDataset
from image import SVHNImage

METADATA_PATH = './data/SVHN/train_metadata.pkl'
IMAGES_PATH = './data/SVHN/train'

def load_pickle(path):
    with open(path, 'rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file

def tensor_to_image(tensor, save=None):
    img = transforms.ToPILImage(mode='RGB')(tensor)
    img.show()
    if save:
        img.save(save)

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

def plot_image_transformation(metadata, data_dir, index=1):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(54),
        transforms.ToTensor(),
    ])

    image_path = '{}/{}.png'.format(data_dir, index)

    image = SVHNImage(metadata[index-1]['metadata'], image_path, crop_percent=0.3, transform=transform)
    orig_image = image.image()
    bounded_image = image.bounded_image()
    cropped_image = image.cropped_image()
    transformed_image = image.transformed_image()

    tensor_to_image(orig_image, save='./figures/original_image.png')
    tensor_to_image(bounded_image, save='./figures/bounded_image.png')
    tensor_to_image(cropped_image, save='./figures/cropped_image.png')
    tensor_to_image(transformed_image, save='./figures/transformed_image.png')

def main(metadata):
    labels = [example['metadata']['label'] for example in metadata.values()]
    lenghts = [len(label) for label in labels]
    digits = [int(digit) for label in labels for digit in label]

    bar_plot(lenghts, xtitle='lenght', title='hist_lenghts')
    bar_plot(digits, xtitle='digit', title='hist_digits')

    

if __name__ == '__main__':
    metadata = load_pickle(METADATA_PATH)
    main(metadata)

    plot_image_transformation(metadata, IMAGES_PATH)
