#!/bin/bash

for j in 1 2 3; do
    for k in 40 64 128 256; do
        for dmodel in 8 16 32; do
            for tp in prefix suffix; do
                sbatch --export=ALL,k=$k,layers=6,batchsize=512,wd=0.01,dmodel=$dmodel,type=$tp launch-copy.slurm
            done
        done 
    done
done