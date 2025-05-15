import os
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()


def aggregate_eval_results(prompt_strategy, threshold):
    """
    Aggregates evaluation results for all benchmarks and model types, selecting samples
    where any model_type has an accuracy drop >= threshold between group_acc and counter_group_acc.

    Args:
        prompt_strategy (str): The prompt strategy to filter on (e.g., "direct_prompting").
        threshold (float): The minimum accuracy drop to consider.
    """
    base_dir = Path(os.getenv("EVAL_HUMAN_ACCEPTED_DIR"))
    if not base_dir.exists():
        print(f"Base directory {base_dir} does not exist.")
        return
    anchor_set = {
        "sample_ids": [],
        "benchmark": [],
    }
    for benchmark_dir in base_dir.iterdir():
        if not benchmark_dir.is_dir():
            continue
        benchmark = benchmark_dir.name
        print(f"\nBenchmark: {benchmark}")
        # Map: sample_id -> set of model_types with sufficient drop
        sample_to_modeltypes = defaultdict(set)
        # Map: sample_id -> dict of model_type -> acc_drop (for those surpassing threshold)
        sample_to_modeltype_accdrop = defaultdict(dict)
        all_model_types = []
        for model_type_dir in benchmark_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            model_type = model_type_dir.name
            all_model_types.append(model_type)
            eval_path = model_type_dir / prompt_strategy / "eval_results.csv"
            if not eval_path.exists():
                continue
            df = pd.read_csv(eval_path)
            for _, row in df.iterrows():
                if type(row["sample_ids"]) == str:
                    sample_id = row["sample_ids"]
                else:
                    sample_id = str(int(row["sample_ids"]))
                group_acc = row["group_acc"]
                counter_group_acc = row["counter_group_acc"]
                acc_drop = counter_group_acc - group_acc
                if acc_drop > threshold:
                    sample_to_modeltypes[sample_id].add(model_type)
                    sample_to_modeltype_accdrop[sample_id][model_type] = acc_drop
        # Output results
        if not sample_to_modeltypes:
            print(
                f"No samples found with accuracy drop > {threshold} for prompt_strategy '{prompt_strategy}'."
            )
            continue
        print(
            f"Samples with accuracy drop > {threshold} for prompt_strategy '{prompt_strategy}':"
        )
        for sample_id, model_types in sample_to_modeltypes.items():
            print(
                f"  Sample ID: {sample_id} | Model Types: {', '.join(sorted(model_types))}"
            )
            # Print accuracy drop by each model type
            for mt in sorted(model_types):
                acc_drop_val = sample_to_modeltype_accdrop[sample_id][mt]
                print(f"    - {mt}: accuracy drop = {acc_drop_val:.3f}")
            anchor_set["sample_ids"].append(sample_id)
            anchor_set["benchmark"].append(benchmark)
    anchor_set_df = pd.DataFrame(anchor_set)
    anchor_set_df.to_csv(base_dir / f"anchor_set.csv", index=False)
    # Ensure sample_ids are strings for comparison
    reference_set = pd.read_csv(base_dir / "reference_anchor_set.csv")
    anchor_sample_ids = set(anchor_set_df["sample_ids"].astype(str))
    reference_sample_ids = set(reference_set["sample_ids"].astype(str))

    only_in_anchor = anchor_sample_ids - reference_sample_ids
    only_in_reference = reference_sample_ids - anchor_sample_ids

    if only_in_anchor:
        print("\nSample IDs in anchor set but NOT in reference set:")
        for sid in sorted(only_in_anchor):
            print(f"  {sid}")
    else:
        print("\nAll anchor set sample_ids are present in reference set.")

    if only_in_reference:
        print("\nSample IDs in reference set but NOT in anchor set:")
        for sid in sorted(only_in_reference):
            print(f"  {sid}")
    else:
        print("\nAll reference set sample_ids are present in anchor set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results for counterfactual verification."
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        required=True,
        help="Prompt method (e.g., direct_prompting)",
    )
    parser.add_argument(
        "--threshold", type=float, required=True, help="Minimum accuracy drop threshold"
    )
    args = parser.parse_args()
    aggregate_eval_results(args.prompt_strategy, args.threshold)
