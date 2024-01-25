#!/bin/bash

for k in 13 14 15 16 17 18 19 20 21 23 24 25; do
    for n in 1 2 4; do
        for g in A5 Z60 A4_x_Z5; do
            sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01,samples=10000 launch-ssm.slurm
        done 
    done
done