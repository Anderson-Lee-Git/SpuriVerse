#!/bin/bash

# Display the script's PID
echo "Script running with PID: $$"

# Array of 5 random seeds
MODELS=("llama" "qwen")
SEEDS=(42 123 7890 54321 9876)
VERSION="v3"
NUM_EPOCHS=10

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running with seed: $SEED"
        # Finetune with anchor data
        echo "Running anchor finetuning..."
        python3 finetuning/finetune.py --model-type $MODEL --layer "both" --data-type "anchor" --seed $SEED --num-epochs $NUM_EPOCHS --use-wandb --wandb-project-name "generalization-$MODEL-$VERSION";

        # Finetune with spurious group data
        echo "Running spurious group finetuning..."
        python3 finetuning/finetune.py --model-type $MODEL --layer "both" --data-type "group" --seed $SEED --num-epochs $NUM_EPOCHS --use-wandb --wandb-project-name "generalization-$MODEL-$VERSION";

        # Finetune with non-spurious data
        echo "Running non-spurious finetuning..."
        python3 finetuning/finetune.py --model-type $MODEL --layer "both" --data-type "non_spurious" --seed $SEED --num-epochs $NUM_EPOCHS --use-wandb --wandb-project-name "generalization-$MODEL-$VERSION";

        # Finetune with mixed data
        echo "Running mixed finetuning..."
        python3 finetuning/finetune.py --model-type $MODEL --layer "both" --data-type "mixed" --seed $SEED --num-epochs $NUM_EPOCHS --use-wandb --wandb-project-name "generalization-$MODEL-$VERSION";
        
        echo "Completed all training runs for seed: $SEED"
        echo "----------------------------------------"
    done
done