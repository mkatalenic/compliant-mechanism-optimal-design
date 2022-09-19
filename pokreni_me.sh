#!/usr/bin/env sh

#SBATCH --job-name=GGS_$2_$3
#SBATCH --partition=computes_thin
#SBATCH -n 1
#SBATCH --ntasks=48

no_divisions=$2
devider=$3

str_dev='_'

out_name=$1$str_dev$2$str_dev$3

mkdir $out_name
cp -r $1/* $out_name
cd $out_name || return

python optimization_ggs.py $no_divisions $devider

rm -r optimization_ggs.py

cd ..

ext='.tar.gz'
tar -cpzf $out_name$ext $out_name
