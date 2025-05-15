# SpuriVerse
This is the official repository for Escaping the SpuriVerse: Can Large Vision-Language Models Generalize Beyond Seen Spurious Correlations?

## Dependencies
```bash
pip install -r requirements.txt
```
## Environment Variables
The repository heavily relies on environment variables. Please set the following in the `.env` file at the root of the repository. The pre-filled values are for relative directory paths. You don't have to change them if you run the commands from the root directory.
```bash
# API Keys
STABILITY_KEY=
OPENAI_KEY=
GEMINI_KEY=
ANTHROPIC_API_KEY=
DASHSCOPE_API_KEY=
# Paths (Make sure to run command from the run directory: SpuriVerse)
AOKVQA_ROOT= # root to aokvqa dataset
AOKVQA_METADATA_PATH="meta/aokvqa_train_meta.csv"
SEEDBENCH_IMG_METADATA_PATH="meta/seedbench_meta.csv"
SEEDBENCH2_METADATA_PATH="meta/seedbench_2_meta.csv"
NATURALBENCH_METADATA_PATH="meta/naturalbench_meta.csv"
# Eval Results Output Directory
FULL_EVAL_RESULTS_DIR="output/benchmark_eval_results"
VLM_ACCEPTED_DIR="output/vlm_accepted"
HUMAN_ACCEPTED_DIR="output/human_accepted"
EVAL_HUMAN_ACCEPTED_DIR="output/eval_human_accepted"
ADHOC_EVAL_RESULTS_DIR="output/adhoc_eval_results"
GROUP_RESPONSES_PATH="output/group_responses"
GROUP_GENERATION_PROD_PATH="output/group_generation_prod"
GROUP_GENERATION_EXPERIMENT_PATH="output/group_generation_experiment"
# Data
LMU_DATA_ROOT= # root to lmu data
# Finetune & Cache
FINETUNE_OUTPUT_DIR="outputs"
MODEL_CACHE_DIR= # cache directory for unsloth models
```
## Download the Dataset
Depending on the components of this repository that you want to use, you may need to download the full benchmarks we used in this repository. Particularly, there are two levels of downloading:
1. The SpuriVerse artifact
2. The full benchmarks required for evaluation on `non_spurious` samples, ablation study, and the entire curation process

For convenience, we also provide the cleaned metadata for the full benchmarks in the `meta` directory, with their annotations from our curation pipeline. These metadata files are the source of question-answer used in this repository.

### SpuriVerse Artifact
This includes all the images in the anchor set, spurious group images, and their questions and answers. It will be sufficient if you would like to
1. evaluate the performance of a VLM on SpuriVerse
2. finetune a VLM on SpuriVerse's anchor or spurious group images

To download the SpuriVerse artifact, run the script:
```bash
python3 adhoc/prepare_data.py
```

### Full Benchmarks
Some components in the experiments require the access to all samples in the original benchmarks, including:
1. evaluation on `non_spurious` samples, which were drawn from the original benchmarks
2. curation of original benchmark samples
3. finetuning a VLM on `non_spurious` samples from the benchmark (ablation study)

If your use case includes these components, you will need to download the full benchmarks by the following instructions.

1. AOKVQA
    We use the official [repository](https://github.com/allenai/aokvqa) to download the dataset. Once you have the dataset downloaded per the instruction of the original repository, you can set the `AOKVQA_ROOT` environment variable to the root of the downloaded dataset, specifically the `aokvqa` directory.
2. SeedBench, SeedBench-2, and NaturalBench
    We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as a shortcut to download these benchmarks. Specifically, you can run evaluations on those benchmarks by following the instruction of the original repository. At the beginning of the evaluation, it'll attempt to download the dataset under a root `LMUData`. You should set the `LMU_DATA_ROOT` environment variable after the download is complete, specifically the `LMUData` directory.

Once you have completed the above steps, you can access the full benchmarks through the `utils` we provided in the repository.

## Usage

### Curation
The `curation` directory contains the code for curating SpuriVerse.
1. `extract_error_set.py` is used to first extract the error set as a challenging set
2. `vlm_filter.py` is then used to produce *VLM accepted* subset
3. `human_verification.py` is a `streamlit` app for human annotation to verify whether a sample from step 2 is faithful. The verified samples will be saved to `output/human_accepted` directory, in the format of `benchmark/spurious_correlation_candidates.csv`.
4. `evaluate_human_accepted_counterfactual.py` is used to evaluate the performance of strong base VLMs on the *Human Accepted* set.
5. `counterfactual_verification.py` is used to select the final `anchor_set` after the evaluation step 4. For reference, we provide the `reference_anchor_set` in the `output/eval_human_accepted` directory, which is the `anchor_set` we derived.

### Adhoc Evaluation
The `adhoc` directory contains the code for evaluation results and prompting experiments.
1. `evaluate.py` is used to evaluate the performance of a VLM on SpuriVerse's different components such as `anchor_set`, `spurious_group`, and `non_spurious` (from original benchmarks).
2. `scripts/prompting_experiment.sh` varies the prompt strategy for models included in the experiments in the paper.
3. `scripts/main_result.sh` produces a table shown in the paper consisting of 15 modern LVLMs' performance on SpuriVerse.

### Finetune
The `finetune` directory contains the code for finetuning a VLM on a subset of SpuriVerse and evaluate its performance on the held-out set of SpuriVerse.
1. `finetune.py` is the main script for finetuning, consisting of various args to configure the experiments.
2. `scripts/ablation.sh` experiments with different subsample sizes, which represent the size of spurious samples used for finetuning.
3. `scripts/generalization.sh` experiments with different finetuning strategies, including finetuning on `anchor_set`, `spurious_group`, and `non_spurious` samples.