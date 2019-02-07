#!/bin/bash
#PBS -A colosse-users
#PBS -l feature=k80
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00  

# PROJECT_PATH will be changed to the master branch of your repo
# PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/'
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/code/'

# RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/evaluation'
RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut1/code/results/'

# DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/data/SVHN/test_sample'
DATA_DIR='/home/user54/digit-detection/data/SVHN/train/'

# METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/data/SVHN/test_sample_metadata.pkl'
METADATA_FILENAME='/home/user54/digit-detection/data/SVHN/train_metadata.pkl'

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

#s_exec python main.py ./config/resnet01.ini
# evaluation/run_evaluation.sh

s_exec python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME