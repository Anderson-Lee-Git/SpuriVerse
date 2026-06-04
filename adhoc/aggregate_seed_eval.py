import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import glob
import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import SUPPORTED_MODELS
from utils.evaluation import compute_acc

load_dotenv()


def aggregate_seed_results(model_type, prompt_strategy, data_type):
    """
    Aggregate evaluation results across different seeds for a given model, prompt strategy, and data type.

    Args:
        model_type (str): The model type (e.g., 'gpt-4o', 'llava-1.6')
        prompt_strategy (str): The prompting strategy (e.g., 'direct_prompting')
        data_type (str): The data type ('anchor', 'spurious_group', or 'non_spurious')

    Returns:
        pd.DataFrame: Aggregated results with mean and std across seeds
    """
    results_dir = (
        Path(os.getenv("ADHOC_EVAL_RESULTS_DIR")) / model_type / prompt_strategy
    )

    # Find all seed result files for the given data type
    pattern = f"{data_type}_seed_*_eval_results.csv"
    seed_files = glob.glob(str(results_dir / pattern))

    if not seed_files:
        print(f"No seed files found for {model_type}/{prompt_strategy}/{data_type}")
        return None

    print(
        f"Found {len(seed_files)} seed files for {model_type}/{prompt_strategy}/{data_type}"
    )

    # Load all seed results
    seed_results = []
    seed_accuracies = []

    for seed_file in seed_files:
        df = pd.read_csv(seed_file)
        seed_results.append(df)

        # Calculate accuracy for this seed
        acc = compute_acc(df)
        seed_accuracies.append(acc)

        # Extract seed number from filename
        seed_num = seed_file.split("seed_")[1].split("_")[0]
        # print(f"Seed {seed_num}: {acc:.4f} accuracy")

    # Calculate mean and std accuracy across seeds
    mean_acc = np.mean(seed_accuracies)
    std_acc = np.std(seed_accuracies)

    # print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    # Create aggregated results DataFrame
    # For now, we'll just create a summary with accuracy statistics
    # You could extend this to aggregate other metrics if needed
    aggregated_results = pd.DataFrame(
        {
            "model_type": [model_type],
            "prompt_strategy": [prompt_strategy],
            "data_type": [data_type],
            "num_seeds": [len(seed_files)],
            "mean_accuracy": [mean_acc],
            "std_accuracy": [std_acc],
            "seed_accuracies": [seed_accuracies],
        }
    )

    return aggregated_results


def main(args):
    """
    Aggregate evaluation results across seeds for specified configurations.
    """
    results = []

    # Process each requested data type
    if args.anchor:
        result = aggregate_seed_results(args.model_type, args.prompt_strategy, "anchor")
        if result is not None:
            results.append(result)

    if args.spurious_group:
        result = aggregate_seed_results(
            args.model_type, args.prompt_strategy, "spurious_group"
        )
        if result is not None:
            results.append(result)

    if args.non_spurious:
        result = aggregate_seed_results(
            args.model_type, args.prompt_strategy, "non_spurious"
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No results to aggregate.")
        return

    # Combine all results
    combined_results = pd.concat(results, ignore_index=True)

    # Save aggregated results
    save_dir = (
        Path(os.getenv("ADHOC_EVAL_RESULTS_DIR"))
        / args.model_type
        / args.prompt_strategy
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    output_file = save_dir / "aggregated_seed_results.csv"
    combined_results.to_csv(output_file, index=False)

    print(f"\nAggregated results saved to: {output_file}")
    print("\nSummary:")
    for _, row in combined_results.iterrows():
        print(
            f"{row['data_type']}: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f} (n={row['num_seeds']})"
        )
    print(
        f"Delta Non-Spurious - Anchor: {combined_results.loc[combined_results['data_type'] == 'non_spurious', 'mean_accuracy'].values[0] - combined_results.loc[combined_results['data_type'] == 'anchor', 'mean_accuracy'].values[0]:.4f}"
    )
    print(
        f"Delta Non-Spurious - Spurious Group: {combined_results.loc[combined_results['data_type'] == 'non_spurious', 'mean_accuracy'].values[0] - combined_results.loc[combined_results['data_type'] == 'spurious_group', 'mean_accuracy'].values[0]:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results across different seeds"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="gpt-4o",
        choices=SUPPORTED_MODELS,
        help="type of the model",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="direct_prompting",
        help="prompting strategy",
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="aggregate anchor set results",
    )
    parser.add_argument(
        "--spurious-group",
        action="store_true",
        help="aggregate spurious group set results",
    )
    parser.add_argument(
        "--non-spurious",
        action="store_true",
        help="aggregate non-spurious set results",
    )
    args = parser.parse_args()
    main(args)
