"""
Module that contains the dataset class for PyTorch
"""
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SVHNDataset(Dataset):
    """
    Class that contains all the functionalities to load the images
    from the SVHN dataset and apply the necessary preprocessing.
    """
    def __init__(self, metadata_path, data_dir, crop_percent=None,
                 transform=None):
        """
        Loads the metadata file.

        :param metadata_path: path to the metadata file.
        :param data_dir: path to the data.
        :param crop_percent: percent to crop taken from
        Goodfellow et al 2013 (30%).
        :param transform: the transforms to apply to each image.
        """
        self._crop_percent = crop_percent
        self._metadata = self._load_pickle(metadata_path)
        self._data_dir = data_dir
        self._transform = transform
        self._img_keys = list(self._metadata.keys())

    def __getitem__(self, index):
        """
        Load the image corresponding to the index and applies
        the transformations.

        :param index: index of the image starting from 0.
        :return: a tuple of the image and the number of digits in it.
        """
        index = self._img_keys[index]
        img_name = '{}/{}.png'.format(self._data_dir, index+1)
        meta = self._metadata[index]['metadata']
        labels = meta['label']
        left = min(meta['left'])
        right = max(meta['left']) + max(meta['width'])
        lower = min(meta['top'])
        upper = max(meta['top']) + max(meta['height'])
        
        img = Image.open(img_name)
        if self._crop_percent:
            img = self._crop(img, left, lower, right, upper)
        img = self._transform(img) if self._transform else np.array(img)
        
        n_digits = len(labels) if len(labels) <= 5 else 6
        return img, n_digits-1

    def __len__(self):
        """
        Return the number of images describe by the metadata file.

        :return: the number of images in the metadata file.
        """
        return len(self._metadata)

    def _load_pickle(self, path):
        """
        Parse the metadata file

        :param path: path to the metadata file
        :return: the metadata file
        """
        with open(path, 'rb') as f:
            pickle_file = pkl.load(f)
        return pickle_file

    def _crop(self, image, left, upper, right, lower):
        """
        Crop an Pil image arround the bounding box that contains all
        the digits and expand the box by 30%

        :param image: the image to crop
        :param left: left bound of the digits
        :param upper: upper bound of the digits
        :param right: right bound of the digits
        :param lower: lower bound of the digits
        :return: the image cropped
        """
        image = image.crop((
            (1 - self._crop_percent) * left,
            (1 - self._crop_percent) * lower,
            (1 + self._crop_percent) * right,
            (1 + self._crop_percent) * upper))
        return image
