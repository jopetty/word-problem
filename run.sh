#!/bin/bash

for k in 17 18 19; do
    for n in 3 4 5; do
        for g in A5; do
            sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01,samples=100000,wsharing=True launch.slurm
        done 
    done
done