#!/usr/bin/env sh


for ((i=1; i<=$no_divisions; i+=1))
do
    python optimization_ggs.py $no_divisions $i
    ime='proba_'
    ext='.tar.gz'
    tar -cpzf $ime$i$ext "probni"
done
