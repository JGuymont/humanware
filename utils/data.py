import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import pickle as pkl
from skimage import io
import numpy as np
from PIL import Image


class SVHNDataset(Dataset):

    def __init__(self, metadata_path, data_dir, crop_percent, transform=None):
        self._crop_percent = crop_percent
        self._metadata = self._load_pickle(metadata_path)
        self._data_dir = data_dir
        self._transform = transform
        self._img_keys = list(self._metadata.keys())

    def __getitem__(self, index):
        index = self._img_keys[index]
        img_name = '{}/{}.png'.format(self._data_dir, index+1)
        meta = self._metadata[index]['metadata']
        labels = meta['label']
        min_left = min(meta['left'])
        max_left = max(meta['left']) + max(meta['width'])
        min_top = min(meta['top'])
        max_top = max(meta['top']) + max(meta['height'])
        
        img = Image.open(img_name)
        img = self._crop(img, min_left, min_top, max_left, max_top)
        img = self._transform(img) if self._transform else np.array(img)
        
        n_digits = len(labels) if len(labels) <= 5 else 5
        return img, n_digits

    def __len__(self):
        return len(self._metadata)

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            pickle_file = pkl.load(f)
        return pickle_file

    def _crop(self, image, min_left, min_top, max_left, max_top):
        """
        cropping pil image according to Goodfellow et al 2013
        """
        image = image.crop((
            (1 - self._crop_percent) * min_left, 
            (1 - self._crop_percent) * min_top,
            (1 + self._crop_percent) * max_left, 
            (1 + self._crop_percent) * max_top))
        return image
