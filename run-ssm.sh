#!/bin/bash

for k in 13 14 15 20; do
    for n in 2 3; do
        for g in A5 A4_x_Z5 Z60; do
            sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01 launch-ssm.slurm
        done 
    done
done