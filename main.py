import os
import argparse
import torch
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data import SVHNDataset
from utils import visualization
from trainer import Trainer
from configparser import ConfigParser

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Split metadata into train/valid/test')
    parser.add_argument('config', type=str)
    parser.add_argument('--train_metadata_path', type=str, default='./data/SVHN/metadata/train_metadata.pkl')
    parser.add_argument('--valid_metadata_path', type=str, default='./data/SVHN/metadata/valid_metadata.pkl')
    parser.add_argument('--test_metadata_path', type=str, default='./data/SVHN/metadata/test_metadata.pkl')
    parser.add_argument('--data_dir', type=str, default='./data/SVHN/train')
    parser.add_argument('--train_pct', type=float, default=0.7)
    parser.add_argument('--valid_pct', type=float, default=0.2)
    parser.add_argument('--test_pct', type=float, default=0.1)

    parser.add_argument('--model', type=str, choices=['SmallCNN', 'MediumCNN', 'LargeCNN', 'ResNet'])
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--crop_percent', type=float, default=0.3)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--device', type=str, default='cpu')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    conf = ConfigParser()
    conf.read(args.config)
    conf.set('model', 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomApply([
        #    transforms.RandomAffine(degrees=conf.getint("randomAffine_degrees"),
        #                            shear=conf.getint("randomAffine_shear")),
        #    transforms.ColorJitter(brightness=conf.getfloat("colorJitter_brightness"),
        #                           contrast=conf.getfloat("colorJitter_contrast"),
        #                           saturation=conf.getfloat("colorJitter_saturation")),
        #    transforms.RandomRotation(conf.getint("randomRotation_degrees")),
        # ], p=0.5),
        transforms.RandomCrop(54),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.39954964, 0.3988817, 0.41280591], [0.23269807, 0.2355513, 0.23580605])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.39954964, 0.3988817, 0.41280591], [0.23269807, 0.2355513, 0.23580605])
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

    
    trainloader = DataLoader(train_data, batch_size=conf.getint("model", "batch_size"), shuffle=True, num_workers=4, pin_memory=True)
    devloader = DataLoader(valid_data, batch_size=100, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=100, num_workers=4, pin_memory=True)

    os.makedirs('results', exist_ok=True)

    args.checkpoints_path = os.path.join(conf.get("model", "checkpoints_path"), conf.get("model", "name"),
                                         datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(conf.get("model", "checkpoints_path"), exist_ok=True)

    trainer = Trainer(conf)
    trainer.train_model(trainloader, devloader)
