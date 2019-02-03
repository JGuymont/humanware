#!/bin/bash
#PBS -A colosse-users
#PBS -l advres=MILA2019
#PBS -l feature=k80
#PBS -l nodes=1:gpus=1
#PBS -l walltime=06:00:00  

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

s_exec python main.py ./config/resnet02.ini
