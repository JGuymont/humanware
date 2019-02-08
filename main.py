"""
Starting point for training the models
"""
import os
import argparse
from datetime import datetime
from configparser import ConfigParser

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data import SVHNDataset
from trainer import Trainer


def argparser():
    """
    Configure the command-line arguments parser

    :return: the arguments parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    conf = ConfigParser()
    conf.read(args.config)
    conf.set('model', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    input_resize = conf.getint("preprocessing", "resize")
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(54),
        transforms.Resize((input_resize, input_resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.39954964, 0.3988817, 0.41280591],
                             [0.23269807, 0.2355513, 0.23580605])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((input_resize, input_resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.39954964, 0.3988817, 0.41280591],
                             [0.23269807, 0.2355513, 0.23580605])
    ])

    train_data = SVHNDataset(
        metadata_path=conf.get("paths", "train_metadata"),
        data_dir=conf.get("paths", "data_dir"),
        crop_percent=conf.getfloat("preprocessing", "crop_percent"),
        transform=train_transforms)

    valid_data = SVHNDataset(
        metadata_path=conf.get("paths", "valid_metadata"),
        data_dir=conf.get("paths", "data_dir"),
        crop_percent=conf.getfloat("preprocessing", "crop_percent"),
        transform=test_transforms)

    test_data = SVHNDataset(
        metadata_path=conf.get("paths", "test_metadata"),
        data_dir=conf.get("paths", "data_dir"),
        crop_percent=conf.getfloat("preprocessing", "crop_percent"),
        transform=test_transforms)

    train_loader = DataLoader(train_data,
                              batch_size=conf.getint("model", "batch_size"),
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    dev_loader = DataLoader(valid_data,
                            batch_size=100,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=100,
                             num_workers=4,
                             pin_memory=True)

    datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    conf.set("paths", "results", os.path.join(conf.get("paths", "results"),
                                              conf.get("model", "name"),
                                              datetime_str))
    os.makedirs(conf.get("paths", 'results'), exist_ok=True)

    conf.set("paths", "checkpoints",
             os.path.join(conf.get("paths", "checkpoints"),
                          conf.get("model", "name"),
                          datetime_str))
    os.makedirs(conf.get("paths", "checkpoints"), exist_ok=True)

    trainer = Trainer(conf)
    trainer.train_model(train_loader, dev_loader)
