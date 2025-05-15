#!/bin/bash

# Display the script's PID
echo "Script running with PID: $$"

# Array of 5 random seeds
SEEDS=(42 123 7890 54321 9876)
TRAIN_SUBSAMPLE_SIZES=(0 20 40 60 80 99)
NUM_EPOCHS=10
MODEL_TYPES=("qwen" "llama")
DATATYPE=("group" "anchor")

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "Running with model type: $MODEL_TYPE"
    for DATATYPE in "${DATATYPES[@]}"; do
        if [ "$DATATYPE" = "anchor" ]; then
            SCALE_FACTOR=1
        else
            SCALE_FACTOR=10
        fi
        for SEED in "${SEEDS[@]}"; do
            echo "Running with seed: $SEED"
            for TRAIN_SUBSAMPLE_SIZE in "${TRAIN_SUBSAMPLE_SIZES[@]}"; do
                echo "Running with train_subsample_size: $TRAIN_SUBSAMPLE_SIZE"
                python3 finetuning/finetune.py --model-type "$MODEL_TYPE" --layer "both" --data-type "$DATATYPE" --seed $SEED --train-subsample --train-subsample-size $TRAIN_SUBSAMPLE_SIZE --num-epochs $NUM_EPOCHS --scale-factor $SCALE_FACTOR --use-wandb --wandb-project-name "ablation-$MODEL_TYPE-$DATATYPE"
            done
            echo "Completed all training runs for seed: $SEED"
            echo "----------------------------------------"
        done
    done
    echo "Completed all runs for model type: $MODEL_TYPE"
    echo "========================================"
done