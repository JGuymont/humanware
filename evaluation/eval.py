import argparse
from pathlib import Path
import sys

sys.path.append('..')

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from trainer import Trainer
from configparser import ConfigParser
from utils.data import SVHNDataset

def eval_model(dataset_dir, metadata_filename, model_filename):

    '''
    Skeleton for your testing function. Modify/add
    all arguments you will need.
    '''

    conf = ConfigParser()
    conf.read('/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/code/config/senet.ini')
    conf.set('model', 'checkpoint', model_filename)
    conf.set('model', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.39954964, 0.3988817, 0.41280591],
                             [0.23269807, 0.2355513, 0.23580605])
    ])

    test_data = SVHNDataset(
        metadata_path=metadata_filename,
        data_dir=dataset_dir,
        crop_percent=conf.getfloat("preprocessing", "crop_percent"),
        transform=test_transforms)

    test_loader = DataLoader(test_data,
                             batch_size=32,
                             num_workers=0,
                             pin_memory=True)

    t = Trainer(conf)
    y_pred = t.make_predictions(test_loader)

    return y_pred


if __name__ == "__main__":

    ###### DO NOT MODIFY THIS SECTION ######
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_filename", type=str, default='')
    # metadata_filename will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--dataset_dir", type=str, default='')
    # dataset_dir will be the absolute path to the directory to be used for
    # evaluation.

    parser.add_argument("--results_dir", type=str, default='')
    # results_dir will be the absolute path to a directory where the output of
    # your inference will be saved.

    args = parser.parse_args()
    metadata_filename = args.metadata_filename
    dataset_dir = args.dataset_dir
    results_dir = args.results_dir
    #########################################


    ###### MODIFY THIS SECTION ######
    # Put your group name here
    group_name = "b1phut1"

    model_filename = '/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/model/senet_model.pth'
    # model_filename should be the absolute path on shared disk to your
    # best model. You need to ensure that they are available to evaluators on
    # Helios.

    #################################


    ###### DO NOT MODIFY THIS SECTION ######
    print("\nEvaluating results ... ")
    y_pred = eval_model(dataset_dir, metadata_filename, model_filename)

    assert type(y_pred) is np.ndarray, "Return a numpy array of dim=1"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = Path(results_dir) / (group_name + '_eval_pred.txt')

    print('\nSaving results to ', results_fname.absolute())
    np.savetxt(results_fname, y_pred, fmt='%.1f')
    #########################################
