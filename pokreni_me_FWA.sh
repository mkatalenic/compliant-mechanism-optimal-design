#!/usr/bin/env sh

<<<<<<< HEAD
#SBATCH --job-name=FWA_kljesta
=======
#SBATCH --job-name=PSO_kljesta
>>>>>>> de028e5286366b5f95380719159af10ebdfd1db4
#SBATCH --partition=computes_thin
#SBATCH -n 1
#SBATCH --ntasks=48

<<<<<<< HEAD
out_name='kljesta_FWA'
cd $out_name || exit

python optimization.py
=======
out_name='kljesta_PSO'
cd $out_name || exit

<<<<<<<< HEAD:pokreni_me_PSO.sh
python optimization.py
========
python3.9 optimization_PSO_from_ggs.py
>>>>>>>> de028e5286366b5f95380719159af10ebdfd1db4:pokreni_me_FWA.sh
>>>>>>> de028e5286366b5f95380719159af10ebdfd1db4

cd ..

ext='.tar.gz'
tar -cpzf $out_name$ext
