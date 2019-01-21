import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data import SVHNDataset
from utils import visualization
from trainer import Trainer

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Split metadata into train/valid/test')
    parser.add_argument('--train_metadata_path', type=str, default='./data/SVHN/metadata/train_metadata.pkl')
    parser.add_argument('--valid_metadata_path', type=str, default='./data/SVHN/metadata/valid_metadata.pkl')
    parser.add_argument('--test_metadata_path', type=str, default='./data/SVHN/metadata/test_metadata.pkl')
    parser.add_argument('--data_dir', type=str, default='./data/SVHN/train')
    parser.add_argument('--train_pct', type=float, default=0.7)
    parser.add_argument('--valid_pct', type=float, default=0.2)
    parser.add_argument('--test_pct', type=float, default=0.1)

    parser.add_argument('--model', type=str, choices=['ModelPaper', 'ConvNet'])
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
    
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.RandomApply([
        #    transforms.RandomAffine(degrees=30, shear=20),
        #    transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=.5),
        #    transforms.RandomRotation(20),
        #], p=0.5),
        transforms.RandomResizedCrop(54),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((54, 54)),
        transforms.ToTensor(),
    ])

    train_data = SVHNDataset(
        metadata_path=args.train_metadata_path, 
        data_dir=args.data_dir, 
        crop_percent=args.crop_percent, 
        transform=train_transforms)

    valid_data = SVHNDataset(
        metadata_path=args.valid_metadata_path, 
        data_dir=args.data_dir, 
        crop_percent=args.crop_percent, 
        transform=test_transforms)
    
    test_data = SVHNDataset(
        metadata_path=args.test_metadata_path, 
        data_dir=args.data_dir, 
        crop_percent=args.crop_percent, 
        transform=test_transforms)

    
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    devloader = DataLoader(valid_data, batch_size=100, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_data, batch_size=100, num_workers=4, pin_memory=True)

    if not os.path.isdir('results'):
        os.mkdir('results')

    trainer = Trainer(args)
    trainer.train_model(trainloader, devloader)
