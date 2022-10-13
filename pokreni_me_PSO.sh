#!/usr/bin/env sh

#SBATCH --job-name=PSO_kljesta
#SBATCH --partition=computes_thin
#SBATCH -n 1
#SBATCH --ntasks=48

out_name='kljesta_PSO'
cd $out_name || exit

python3.9 optimization.py

cd ..

ext='.tar.gz'
tar -cpzf $out_name$ext
