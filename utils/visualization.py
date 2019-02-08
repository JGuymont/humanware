"""
Module containing functions for visualizing the data
"""
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms

from image import SVHNImage

METADATA_PATH = './data/SVHN/train_metadata.pkl'
IMAGES_PATH = './data/SVHN/train'


def load_pickle(path):
    """
    Function to load a metadatafile

    :param path: path to the file
    :return: the metadata
    """
    with open(path, 'rb') as f:
        pickle_file = pickle.load(f)
    return pickle_file


def tensor_to_image(tensor, save=None):
    """
    Function to convert a tensor to a PIL image, show it and save it.

    :param tensor: the image in tensor form
    :param save: whether to save or not the image
    """
    img = transforms.ToPILImage(mode='RGB')(tensor)
    img.show()
    if save:
        img.save(save)


def count_elements(seq) -> dict:
    """
    Return the number of frequency of each element of a list

    :param seq: list of int
    :return: dictionary where the keys are category of 
        element in the list `seq` and the values are 
        the frequency

    Example: Calling count_element([1, 1, 1, 2, 2]) would
        return `{1: 3, 2: 2}`  
    """
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


def bar_plot(data, xtitle, title):
    """
    Function to create the plot and save it.

    :param data: data to plot
    :param xtitle: name of the x axis
    :param title: name of the plot
    """
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
    """
    Function to produce all the stage of the preprocessing for an input image.

    :param metadata: metadata of the images
    :param data_dir: path to the images
    :param index: index of the image
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(54),
        transforms.ToTensor(),
    ])

    image_path = '{}/{}.png'.format(data_dir, index)

    image = SVHNImage(metadata[index-1]['metadata'], image_path,
                      crop_percent=0.3, transform=transform)
    orig_image = image.image()
    bounded_image = image.bounded_image()
    cropped_image = image.cropped_image()
    transformed_image = image.transformed_image()

    tensor_to_image(orig_image, save='./figures/original_image.png')
    tensor_to_image(bounded_image, save='./figures/bounded_image.png')
    tensor_to_image(cropped_image, save='./figures/cropped_image.png')
    tensor_to_image(transformed_image, save='./figures/transformed_image.png')

    
if __name__ == '__main__':
    metadata = load_pickle(METADATA_PATH)
    labels = [example['metadata']['label'] for example in metadata.values()]
    lenghts = [len(label) for label in labels]
    digits = [int(digit) for label in labels for digit in label]

    bar_plot(lenghts, xtitle='lenght', title='hist_lenghts')
    bar_plot(digits, xtitle='digit', title='hist_digits')

    plot_image_transformation(metadata, IMAGES_PATH)
