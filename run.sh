#!/bin/bash

for k in 20 20 20; do
    for n in 1 2 3; do
        for g in A5 A4_x_Z5; do
            sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01,samples=1000000,wsharing=False launch.slurm
        done 
    done
done