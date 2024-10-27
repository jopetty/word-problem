#!/bin/bash
# Easily launch finetuning jobs on Gantry.

ROOT=${ROOT:-"/net/nfs.cirrascale/allennlp/willm/log-depth"}
DEPTHS=("$@")
MODEL="sfirah"
WIDTH=512
UNIVERSAL=${UNIVERSAL:-False}
GPUS=${GPUS:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"log-depth"}

for depth in "${DEPTHS[@]}"; do
    run_name="$MODEL-d$depth"
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
            --d-model $WIDTH \
            --d-ff $((WIDTH * 4)) \
            --depth $depth \
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
            --lr-schedule "constant" \
            --universal $UNIVERSAL \
            --wandb-project $WANDB_PROJECT
done