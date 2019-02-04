import os
import argparse
import torch
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data import SVHNDataset
from trainer import Trainer
from configparser import ConfigParser
from models.residual_network import ResNet

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='Split metadata into train/valid/test')
    parser.add_argument('--config', type=str)
    parser.add_argument('--task', type=str, default='train')
    
    return parser.parse_args()

def restore_model(self, epoch):
    """Retore the model parameters

    Args
        epoch: (int) epoch at which the model has been 
            trained and for which the model paramers
            should be restored
    """
    path = '{}{}.pt'.format(self.config['model']['path'], epoch)
    checkpoint = torch.load(path)
    self.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

if __name__ == '__main__':
    args = argparser()
    conf = ConfigParser()
    conf.read(args.config)
    conf.set('model', 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.RandomApply([
        #    transforms.RandomAffine(degrees=conf.getint("preprocessing", "randomAffine_degrees"),
        #                            shear=conf.getint("preprocessing", "randomAffine_shear")),
        #    transforms.ColorJitter(brightness=conf.getfloat("preprocessing", "colorJitter_brightness"),
        #                           contrast=conf.getfloat("preprocessing", "colorJitter_contrast"),
        #                           saturation=conf.getfloat("preprocessing", "colorJitter_saturation")),
        #    transforms.RandomRotation(conf.getint("preprocessing", "randomRotation_degrees")),
        #], p=conf.getfloat("preprocessing", "transform_proba")),
        transforms.RandomCrop(54),
        transforms.ToTensor(),
        #transforms.Normalize([0.39954964, 0.3988817, 0.41280591], [0.23269807, 0.2355513, 0.23580605])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((54, 54)),
        transforms.ToTensor(),
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

    os.makedirs('results', exist_ok=True)

    

    if args.task == 'train':
        trainloader = DataLoader(
            train_data, 
            batch_size=conf.getint("model", "batch_size"), 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True)
        devloader = DataLoader(valid_data, batch_size=100, num_workers=4, pin_memory=True)
        
        conf.set("model", "checkpoints_path", os.path.join(
            conf.get("model", "checkpoints_path"), 
            conf.get("model", "name"),
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
            
        os.makedirs(conf.get("model", "checkpoints_path"), exist_ok=True)
        
        trainer = Trainer(conf)
        trainer.train_model(trainloader, devloader)
    
    elif args.task == 'eval':
        model = eval(conf['model'].get('name'))(conf['model']).to(conf['model']['device'])
        checkpoint = torch.load(PATH_TO_BEST_CHECKPOINT)
        model.load_state_dict(checkpoint['state_dict'])

        testloader = DataLoader(test_data, batch_size=100, num_workers=4, pin_memory=True)
        model.eval()
        total, correct = 0., 0.
        for (inputs, targets) in testloader:
            inputs = inputs.to(conf['model']['device'])
            targets = targets.to(conf['model']['device'])

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        accuracy = 100. * correct / total
        print(round(accuracy, 4))
        

    
