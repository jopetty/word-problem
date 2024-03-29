#!/bin/bash

for r in 1 1 1; do
    for k in 6 7 8 9 10 11 12 20; do
        for n in 1 2 3; do
            for g in Z60 A4_x_Z5 A5; do
                sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01,samples=1000000,wsharing=False launch-mamba.slurm
            done 
        done
    done
done