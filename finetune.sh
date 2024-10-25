#!/bin/bash

args=("$@")
for arg in "${args[@]}"; do
    echo $arg
done
exit

SIZES=("$@")
ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
SUFFIX=${SUFFIX:""}  # Can set to "-deduped"
GPUS=${GPUS:-1}

mkdir $OUT_DIR/$SAVE
for size in "${SIZES[@]}"; do
    model="pythia-$size$SUFFIX"
    echo "===== $model ====="
    printf "$model" | gantry run \
        --workspace ai2/rusty-dawg \
        --cluster ai2/allennlp-cirrascale \
        --budget ai2/allennlp \
        --priority normal \
        --env-secret "WANDB_API_KEY=WANDB_API_KEY" \
        --gpus $GPUS -- python src/finetune.py \
            --model "EleutherAI/$model" \
            --train-paths \
                $ROOT/data/2/train.csv \
                $ROOT/data/4/train.csv \
                $ROOT/data/8/train.csv \
                $ROOT/data/16/train.csv \
                $ROOT/data/32/train.csv \
                $ROOT/data/64/train.csv \
                $ROOT/data/128/train.csv \
            --eval-path $ROOT/data/128/val.csv \
            --results-dir $ROOT/checkpoints/$model \
            --logs-dir $ROOT/checkpoints/$model/logs \
            --batch-size 64 \
            --warmup-steps 500 \
            --log-steps 100 \
            --eval-steps 100 \
            --indices 0 1 3 7 15 31 63 127 \
            --lr-schedule "constant"
done