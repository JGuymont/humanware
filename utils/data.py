import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import pickle as pkl
from skimage import io
import numpy as np
from PIL import Image


class SVHNDataset(Dataset):

    def __init__(self, metadata_path, data_dir, transform=None, root=None):
        
        self._metadata = self._load_pickle(metadata_path)
        self.filenames = [meta['filename'] for meta in self._metadata.values()] 
        self._data_dir = data_dir
        self.transform = transform

    def __getitem__(self, index):
        img_name = '{}/{}.png'.format(self._data_dir, index+1)
        meta = self._metadata[index]['metadata']
        height = meta['height']
        left = meta['left']
        top = meta['top']
        width = meta['width']
        label = meta['label']
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)
        n_digits = len(label) if len(label) <= 5 else 5
        return img, n_digits

    def __len__(self):
        return len(self.filenames)

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            pickle_file = pkl.load(f)
        return pickle_file
