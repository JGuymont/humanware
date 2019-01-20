import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from utils.data import SVHNDataset
from utils import visualization
from models.cnn import ConvNet

TRAIN_METADATA_PATH = './data/SVHN/train_metadata.pkl'
TRAIN_DATA_DIR = './data/SVHN/train'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
# TODO: Put in a config file
num_epochs = 5
num_classes = 5
batch_size = 100
learning_rate = 0.001

if __name__ == '__main__':
    

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = SVHNDataset(metadata_path=TRAIN_METADATA_PATH, data_dir=TRAIN_DATA_DIR, transform=train_transforms)
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(trainloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, i+1, total_step, loss.item()))

    