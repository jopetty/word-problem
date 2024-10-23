#!/bin/bash

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
SUFFIX=${SUFFIX:""}  # Can set to "-deduped"
# SIZES=("70m" "160m" "410m" "1b" "1.4b" "2.8b" "6.9b" "12b")
# SIZES=("70m" "160m" "410m")
SIZES=("70m")

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
        --gpus 1 -- python src/finetune.py \
            --model "EleutherAI/$model" \
            --phase-name \
                length2 \
                length4 \
                length128 \
            --train-path \
                $ROOT/data/2/train.csv \
                $ROOT/data/4/train.csv \
                $ROOT/data/128/train.csv \
            --val-path \
                $ROOT/data/128/val.csv \
                $ROOT/data/128/val.csv \
                $ROOT/data/128/val.csv \
            --results-dir $ROOT/checkpoints/$model \
            --logs-dir $ROOT/checkpoints/$model/logs \
            --batch-size 64 \
            --warmup-steps 500 \
            --log-steps 100 \
            --eval-steps 100 \
            --indices 0 1 2 4 8 16 32 64 127
done