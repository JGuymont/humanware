"""
Split metadata into train/valid/test
"""
import argparse
import os

import pickle


def argparser():
    """
    Configure the command-line arguments parser

    :return: the arguments parsed
    """
    parser = argparse.ArgumentParser(
        description='Split metadata into train/valid/test'
    )
    parser.add_argument('--metadata_path',
                        type=str,
                        default='./data/SVHN/train_metadata.pkl')
    parser.add_argument('--metadata_dir',
                        type=str,
                        default='./data/SVHN/metadata')
    parser.add_argument('--train_pct', type=float, default=0.7)
    parser.add_argument('--valid_pct', type=float, default=0.2)
    parser.add_argument('--test_pct', type=float, default=0.1)
    return parser.parse_args()


def save_metadata(metadata, path):
    """
    Create a pickle file for the metadata

    :param metadata: the metadata file to save
    :param path: where to save the file
    """
    with open(path, 'wb') as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def main(args):
    """
    Split a metadata file in train/valid/test metadata files

    :param args: Command-line args to parse
    """
    with open(args.metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    assert round((args.train_pct
                  + args.valid_pct
                  + args.test_pct) * 100) / 100 == 1
    data_size = len(metadata)
    train_idx = range(int(args.train_pct * data_size))
    valid_idx = range(int(args.train_pct * data_size),
                      int((args.train_pct + args.valid_pct) * data_size))
    test_idx = range(int((args.train_pct+args.valid_pct)*data_size), data_size)
    
    train_metadata = {i:metadata[i] for i in train_idx}
    valid_metadata = {i:metadata[i] for i in valid_idx}
    test_metadata = {i:metadata[i] for i in test_idx}

    save_metadata(train_metadata, '{}/{}.pkl'.format(args.metadata_dir,
                                                     'train_metadata'))
    save_metadata(valid_metadata, '{}/{}.pkl'.format(args.metadata_dir,
                                                     'valid_metadata'))
    save_metadata(test_metadata, '{}/{}.pkl'.format(args.metadata_dir,
                                                    'test_metadata'))


if __name__ == '__main__':
    args = argparser()
    if not os.path.isdir(args.metadata_dir):
        os.mkdir(args.metadata_dir)
    main(args)
