#!/usr/bin/env sh

#SBATCH --job-name=GGS_optim
#SBATCH --partition=computes_thin
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=2G

no_divisions=10

for ((i=1; i<=$no_divisions; i+=1))
do
    python3.9 optimization_ggs.py $no_divisions $i
    ime='GGS_'
    ext='.tar.gz'
    tar -cpzf $ime$i$ext "ccx_files" "img" "mesh_setup.pkl"
done
