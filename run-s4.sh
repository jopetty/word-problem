#!/bin/bash

for r in 1 2 3; do
    for k in 20; do
        for n in 3 4 5; do
            for g in A5 A4_x_Z5 Z60; do
                sbatch --export=ALL,k=$k,layers=$n,group=$g,batchsize=512,wd=0.01 launch-s4.slurm
            done 
        done
    done
done