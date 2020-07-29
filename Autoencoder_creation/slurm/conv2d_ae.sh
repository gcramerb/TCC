#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --qos=high

if [ $# -lt 1 ]
  then
    echo "No sufficient arguments supplied (please provide config file, output file and workdir.)"
    exit
fi

DATASET=$1
#conda env update -f environment.yml
source activate pix_hell
#
conda install tensorflow-gpu==1.4.1 
conda install cudnn=7.0.5
pip install keras==2.2

srun python ../conv2d-ae.py $DATASET
