#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
N_TRAIN=${N_TRAIN:-10000000}
N_VAL=${N_VAL:-10000}
MAX_K=${MAX_K:-128}

for ((k=2; k<=MAX_K; k*=2)); do
    mkdir -p $ROOT/data/$k
    python src/generate_data.py A5 --k $k --data-path $ROOT/data/$k/train.csv --samples $N_TRAIN
    python src/generate_data.py A5 --k $k --data-path $ROOT/data/$k/val.csv --samples $N_VAL
done