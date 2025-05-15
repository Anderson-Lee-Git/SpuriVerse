# Run evaluate.py for all supported models in utils/constants.py

SUPPORTED_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gemini-1.5-pro"
    "gemini-2.0-flash"
    "claude-3.7-sonnet"
    "qwen-vl-max"
    "qwen-vl-plus"
    "qwen-2.5-7b"
    "qwen-2.5-32b"
    "llama-3.2-11b"
    "llama-3.2-90b"
    "llava-1.5"
    "llava-1.6"
    "o4-mini"
    "o3"
)

for MODEL in "${SUPPORTED_MODELS[@]}"; do
    echo "Running evaluate.py for model: $MODEL"
    python adhoc/evaluate.py --model-type "$MODEL" --anchor --spurious-group --non-spurious --prompt-strategy "direct_prompting"
done

