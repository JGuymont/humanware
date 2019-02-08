#!/bin/bash


# PROJECT_PATH will be changed to the master branch of your repo
# PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/'
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/code/'

# RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/evaluation'
RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/code/results/'

# DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/data/SVHN/test_sample'
DATA_DIR='/home/user40/Humanware-block1/data/SVHN/train/'

# METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/data/SVHN/test_sample_metadata.pkl'
METADATA_FILENAME='/home/user40/Humanware-block1/data/SVHN/metadata/test_metadata.pkl'

cd $PROJECT_PATH/evaluation
echo $PROJECT_PATH
echo 'Entering eval.py'
s_exec python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME
