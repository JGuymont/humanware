import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import pickle as pkl
from skimage import io
import numpy as np
from PIL import Image


class TrainSVHNDataset(Dataset):
    def __init__(self, transform=None, root=None):
        self.root = root
        self.transforms = transform

        # extract data from pkl file -> dict
        with open('train_metadata.pkl', 'rb') as f:
            train_metadata = pkl.load(f)
        self.train_df = pd.concat({k: pd.DataFrame.from_dict(v).T for k, v in train_metadata.items()})

        # split train/val
        train_percent = 0.7
        val_percent = 0.2
        test_percent = 0.1
        assert round((train_percent + val_percent + test_percent)*100)/100 == 1

        self.train_df_filenames = self.train_df.iloc[::2, :1]
        self.train_df_filenames_train = self.train_df_filenames[:int(train_percent * len(self.train_df_filenames))]
        self.train_df_filenames_val = self.train_df_filenames[int(train_percent * len(self.train_df_filenames)):int(
            (train_percent + val_percent) * len(self.train_df_filenames))]
        self.train_df_filenames_test = self.train_df_filenames[
                                       int((train_percent + val_percent) * len(self.train_df_filenames)):]

        self.train_df_metadata = self.train_df.iloc[1::2, :]
        self.train_df_metadata_train = self.train_df_metadata[:int(train_percent * len(self.train_df_metadata))]
        self.train_df_metadata_val = self.train_df_metadata[int(train_percent * len(self.train_df_metadata)):int(
            (train_percent + val_percent) * len(self.train_df_filenames))]
        self.train_df_metadata_test = self.train_df_metadata[
                                      int((train_percent + val_percent) * len(self.train_df_metadata)):]

        self.train_df_filenames_val.to_pickle("val_filenames.pkl")
        self.train_df_metadata_val.to_pickle('val_metadata.pkl')

        self.train_df_filenames_test.to_pickle("test_filenames.pkl")
        self.train_df_metadata_test.to_pickle("test_metadata.pkl")

    def __getitem__(self, idx):
        # print('train/' + self.train_df_filenames.iloc[idx, 0])

        img_name = 'train/' + self.train_df_filenames_train.iloc[idx, 0]
        height = self.train_df_metadata_train.iloc[idx, 0]
        label = self.train_df_metadata_train.iloc[idx, 1]
        left = self.train_df_metadata_train.iloc[idx, 2]
        top = self.train_df_metadata_train.iloc[idx, 3]
        width = self.train_df_metadata_train.iloc[idx, 4]
        img_tmp = np.array(io.imread(img_name), dtype=np.uint8)
        img_tmp = np.asarray(img_tmp - np.mean(img_tmp, axis=(0, 1), keepdims=True), dtype=np.uint8)
        # print(img_tmp.shape)
        min_left = min(left)
        max_left = max(left) + max(width)

        min_top = min(top)
        max_top = max(top) + max(height)

        img_tmp2 = Image.fromarray(img_tmp)

        # cropping pil image according to Goodfellow et al 2013
        crop_percent = 0.3
        img_tmp2 = img_tmp2.crop(((1 - crop_percent) * min_left, min_top * (1 - crop_percent),
                                  (1 + crop_percent) * max_left, max_top * (1 + crop_percent)))

        # self.plot_img(img_tmp2)
        img = self.transforms(img_tmp2)

        # len(label) is the amount of items in the image
        img_labels = len(label)
        # print(label)
        # print(img, img_labels)
        return img, img_labels

    def __len__(self):
        return len(self.train_df_filenames_train)

    def plot_img(self, img):
        img.show()


class ValSVHNDataset(Dataset):
    def __init__(self, transform=None, root=None):
        self.transforms = transform

        # self.validation_filenames = pd.read_csv('val_filenames.csv', header=None)
        self.validation_filenames = pd.read_pickle('val_filenames.pkl')
        self.validation_metadata = pd.read_pickle('val_metadata.pkl')

    def __getitem__(self, idx):
        img_name = 'train/' + self.validation_filenames.iloc[idx, 0]
        height = self.validation_metadata.iloc[idx, 0]
        label = self.validation_metadata.iloc[idx, 1]
        left = self.validation_metadata.iloc[idx, 2]
        top = self.validation_metadata.iloc[idx, 3]
        width = self.validation_metadata.iloc[idx, 4]
        img_tmp = np.array(io.imread(img_name), dtype=np.uint8)
        img_tmp = np.asarray(img_tmp - np.mean(img_tmp, axis=(0, 1), keepdims=True), dtype=np.uint8)

        min_left = min(left)
        max_left = max(left) + max(width)
        min_top = min(top)
        max_top = max(top) + max(height)
        img_tmp2 = Image.fromarray(img_tmp)

        # cropping pil image according to Goodfellow et al 2013
        crop_percent = 0.3
        img_tmp2 = img_tmp2.crop(((1 - crop_percent) * min_left, min_top * (1 - crop_percent),
                                  (1 + crop_percent) * max_left, max_top * (1 + crop_percent)))

        img = self.transforms(img_tmp2)

        # len(label) is the amount of items in the image
        img_labels = len(label)
        # print(img, img_labels)
        return img, img_labels

    def __len__(self):
        return len(self.validation_filenames)


class SyntheticTestSVHNDataset(Dataset):
    def __init__(self, transform=None, root=None):
        self.transforms = transform

        # self.validation_filenames = pd.read_csv('val_filenames.csv', header=None)
        self.test_filenames = pd.read_pickle('test_filenames.pkl')
        self.test_metadata = pd.read_pickle('test_metadata.pkl')

    def __getitem__(self, idx):
        img_name = 'train/' + self.test_filenames.iloc[idx, 0]
        height = self.test_metadata.iloc[idx, 0]
        label = self.test_metadata.iloc[idx, 1]
        left = self.test_metadata.iloc[idx, 2]
        top = self.test_metadata.iloc[idx, 3]
        width = self.test_metadata.iloc[idx, 4]
        img_tmp = np.array(io.imread(img_name), dtype=np.uint8)
        img_tmp = np.asarray(img_tmp - np.mean(img_tmp, axis=(0, 1), keepdims=True), dtype=np.uint8)

        min_left = min(left)
        max_left = max(left) + max(width)
        min_top = min(top)
        max_top = max(top) + max(height)
        img_tmp2 = Image.fromarray(img_tmp)

        # cropping pil image according to Goodfellow et al 2013
        crop_percent = 0.3
        img_tmp2 = img_tmp2.crop(((1 - crop_percent) * min_left, min_top * (1 - crop_percent),
                                  (1 + crop_percent) * max_left, max_top * (1 + crop_percent)))

        img = self.transforms(img_tmp2)

        # len(label) is the amount of items in the image
        img_labels = len(label)
        # print(img, img_labels)
        return img, img_labels

    def __len__(self):
        return len(self.test_filenames)
