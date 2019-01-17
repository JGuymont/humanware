import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import pickle as pkl
from skimage import io
import numpy as np


class TrainSVHNDataset(Dataset):
    def __init__(self, transform=None, root=None):
        with open('train_metadata.pkl', 'rb') as f:
            train_metadata = pkl.load(f)

        self.train_df = pd.concat({k: pd.DataFrame.from_dict(v).T for k, v in train_metadata.items()})

        #leave the choice to other teams to have other dataframes so that they can analyze other parts of the data
        self.train_df_filenames = self.train_df.iloc[::2, :1]
        self.train_df_metadata = self.train_df.iloc[1::2, :]

        self.transforms = transform
    def __getitem__(self, idx):
        print("get item")
        img_name = 'train/' + self.train_df_filenames[idx, 0]
        print(img_name)
        height = train_df_metadata[idx, 0]
        label = train_df_metadata[idx, 1]
        left = train_df_metadata[idx, 2]
        top = train_df_metadata[idx, 3]
        width = train_df_metadata[idx, 4]

        img = np.array(io.imread(img_name), dtype=np.uint8)
        print(img)
        img_labels = top #is that what we want to analyze?

        return img, img_labels

    def __len__(self):
        return len(self.train_df_filenames)