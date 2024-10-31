#!/bin/bash
# Easily launch finetuning jobs on Gantry.

SIZES=("$@")
ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
SUFFIX=${SUFFIX:""}  # Can set to "-deduped"
GPUS=${GPUS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"log-depth"}
BATCH_SIZE=${BATCH_SIZE:-64}

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
            --run-name "$model" \
            --train-paths \
                $ROOT/data/2/train.csv \
                $ROOT/data/4/train.csv \
                $ROOT/data/8/train.csv \
                $ROOT/data/16/train.csv \
                $ROOT/data/32/train.csv \
                $ROOT/data/64/train.csv \
                $ROOT/data/128/train.csv \
                $ROOT/data/256/train.csv \
                $ROOT/data/512/train.csv \
            --eval-path $ROOT/data/512/val.csv \
            --results-dir $ROOT/checkpoints/$model \
            --logs-dir $ROOT/checkpoints/$model/logs \
            --batch-size $BATCH_SIZE \
            --warmup-steps 500 \
            --log-steps 1000 \
            --eval-steps 1000 \
            --indices 0 1 3 7 15 31 63 127 255 511 \
            --eps 0.05 0.5 \
            --lr-schedule "constant" \
            --wandb-project $WANDB_PROJECT
done