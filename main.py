import pickle as pkl

DATA_PATH = './data/SVHN/train_metadata.pkl'


if __name__ == '__main__':
    with open(DATA_PATH, 'rb') as f:
        train_metadata = pkl.load(f)
    
    print(train_metadata)