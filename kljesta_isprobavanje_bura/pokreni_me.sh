#!/usr/bin/env sh

#SBATCH --job-name=GGS_optim
#SBATCH --partition=computes_thin
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=500M

srun bash GGS.sh
