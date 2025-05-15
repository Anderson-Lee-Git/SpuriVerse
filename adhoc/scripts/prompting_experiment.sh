# Run evaluate.py for all supported models in utils/constants.py

SUPPORTED_MODELS=(
    "qwen-2.5-7b"
    "gpt-4o"
    "llama-3.2-11b"
)

PROMPT_STRATEGIES=(
    "direct_prompting"
    "chain_of_thought"
    "spurious_aware"
)


for MODEL in "${SUPPORTED_MODELS[@]}"; do
    for PROMPT_STRATEGY in "${PROMPT_STRATEGIES[@]}"; do
        echo "Running evaluate.py for model: $MODEL with prompt strategy: $PROMPT_STRATEGY"
        python3 adhoc/evaluate.py --model-type "$MODEL" --anchor --spurious-group --non-spurious --prompt-strategy "$PROMPT_STRATEGY";
    done
done

