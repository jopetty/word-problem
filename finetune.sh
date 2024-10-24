#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
SUFFIX=${SUFFIX:""}  # Can set to "-deduped"
GPUS=1
# SIZES=("14m" "31m" "70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
SIZES=("14m" "31m")

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
            --val-path $ROOT/data/128/val.csv \
            --results-dir $ROOT/checkpoints/$model \
            --logs-dir $ROOT/checkpoints/$model/logs \
            --batch-size 64 \
            --warmup-steps 500 \
            --log-steps 100 \
            --eval-steps 100 \
            --indices 0 1 3 7 15 31 63 127
done