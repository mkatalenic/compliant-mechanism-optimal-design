#!/usr/bin/env sh

#SBATCH --job-name=GGS_kljesta_10
#SBATCH --partition=computes_thin
#SBATCH -n 1
#SBATCH --ntasks=48

out_name='kljesta_GGS_12_4_10'

python3.9 optimization.py $out_name

ext='.tar.gz'
tar -cpzf $out_name$ext
