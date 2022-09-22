#!/usr/bin/env sh

#SBATCH --job-name=PSO_from_GGS
#SBATCH --partition=computes_thin
#SBATCH -n 1
#SBATCH --ntasks=48

out_name='kljesta_6_4'
cd $out_name || exit

python3.9 optimization_PSO_from_ggs.py

cd ..

ext='.tar.gz'
tar -cpzf $out_name$ext
