#!/bin/bash

# Define the available model types and benchmarks
MODEL_TYPES=("gemini-2.0-flash" "gpt-4o" "qwen-vl-max")
BENCHMARKS=("seedbench_img" "seedbench2" "aokvqa" "naturalbench")  # Add more benchmarks here if needed
PROMPT_STRATEGIES=("direct_prompting")

EVAL_SCRIPT="curation/evaluate_human_accepted_counterfactual.py"

for BENCHMARK in "${BENCHMARKS[@]}"; do
  for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for PROMPT_STRATEGY in "${PROMPT_STRATEGIES[@]}"; do
      echo "Running evaluation for model: $MODEL_TYPE, benchmark: $BENCHMARK, prompt strategy: $PROMPT_STRATEGY"
      python3 "$EVAL_SCRIPT" \
        --model-type "$MODEL_TYPE" \
        --benchmark "$BENCHMARK" \
        --prompt-strategy "$PROMPT_STRATEGY"
    done
  done
done
