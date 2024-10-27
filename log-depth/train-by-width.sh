#!/bin/bash
# Easily launch finetuning jobs on Gantry.

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
WIDTHS=("$@")
MODEL="sfirah"
DEPTH=6
UNIVERSAL=${UNIVERSAL:-False}
GPUS=${GPUS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"log-depth"}

for width in "${WIDTHS[@]}"; do
    run_name="$MODEL-w$width"
    if [ "$UNIVERSAL" = True ]; then
        run_name="$run_name-u"
    fi
    echo "===== $run_name ====="
    printf $run_name | gantry run \
        --workspace ai2/rusty-dawg \
        --cluster ai2/allennlp-cirrascale \
        --budget ai2/allennlp \
        --priority normal \
        --env-secret "WANDB_API_KEY=WANDB_API_KEY" \
        --gpus $GPUS -- python src/finetune.py \
            --model "sfirah" \
            --run-name $run_name \
            --d-model $width \
            --d-ff $((width * 4)) \
            --depth $DEPTH \
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
            --log-steps 1000 \
            --eval-steps 1000 \
            --indices 0 1 3 7 15 31 63 127 \
            --eps 0.05 0.5 \
            --lr-schedule "constant" \
            --universal $UNIVERSAL \
            --wandb-project $WANDB_PROJECT
done