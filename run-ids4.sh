#!/bin/bash

for r in 1 2 3; do
    for k in 3 4 5 6 7 8 9 10 11 12 30; do
        for n in 1 2 3 4 5; do
            for g in A5 A4_x_Z5 Z60; do
                sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01 launch-ids4.slurm
            done 
        done
    done
done