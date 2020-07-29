#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=viper05

conda env update -q
source activate harkeraspython36
srun python ../conv2d-ae.py