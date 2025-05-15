import os
import argparse
import logging

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_preparation import (
    get_sample_by_image_id,
    parse_data,
)
from utils.constants import SUPPORTED_BENCHMARKS
from curation.pipeline import Pipeline
from curation.engine import VLMInterface

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def calculate_group_accuracy(image_id, pos_filepath, neg_filepath, benchmark):
    pos_df = pd.read_csv(pos_filepath)
    neg_df = pd.read_csv(neg_filepath)

    sample = get_sample_by_image_id(image_id, benchmark)
    ground_truth = sample["answer"]

    pos_predictions = pos_df["response"].apply(parse_data)
    neg_predictions = neg_df["response"].apply(parse_data)

    pos_results = pos_predictions == ground_truth
    pos_acc = pos_results.sum() / pos_results.shape[0]
    neg_results = neg_predictions == ground_truth
    neg_acc = neg_results.sum() / neg_results.shape[0]
    return pos_acc, neg_acc


def main(args):
    """
    Evaluates models' performance on counterfactual group generation
    based on human-accepted spurious correlation candidates
    """
    pos_results = []
    neg_results = []

    benchmark = args.benchmark
    seed_spurious_correlations_df = pd.read_csv(
        Path(os.getenv("HUMAN_ACCEPTED_DIR"))
        / benchmark
        / "spurious_correlation_candidates.csv"
    )
    sample_ids = seed_spurious_correlations_df["sample_id"].values
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    logger.info(f"Evaluating {len(sample_ids)} samples")
    # evaluate with model
    model = VLMInterface(args.model_type)
    logger.info("Evaluating group responses...")
    for sample_id in tqdm(sample_ids):
        if not args.re_evaluate_group_responses:
            pos_filepath = str(
                Path(os.getenv("GROUP_RESPONSES_PATH"))
                / benchmark
                / args.model_type
                / args.prompt_strategy
                / f"group_responses_{sample_id}.csv"
            )
            neg_filepath = str(
                Path(os.getenv("GROUP_RESPONSES_PATH"))
                / benchmark
                / args.model_type
                / args.prompt_strategy
                / f"group_responses_{sample_id}_counter.csv"
            )
            if os.path.exists(pos_filepath) and os.path.exists(neg_filepath):
                continue
        sample_id = str(sample_id)
        pos_directory = str(
            Path(os.getenv("GROUP_GENERATION_PROD_PATH")) / args.benchmark / sample_id
        )
        neg_directory = str(
            Path(os.getenv("GROUP_GENERATION_PROD_PATH"))
            / args.benchmark
            / f"{sample_id}_counter"
        )
        # evaluate positive group
        Pipeline.evaluate(
            image_id=sample_id,
            directory=pos_directory,
            model=model,
            model_type=args.model_type,
            benchmark=args.benchmark,
            prompt_strategy=args.prompt_strategy,
        )
        # evaluate negative group
        Pipeline.evaluate(
            image_id=sample_id,
            directory=neg_directory,
            model=model,
            model_type=args.model_type,
            benchmark=args.benchmark,
            prompt_strategy=args.prompt_strategy,
        )
        # The results will be saved in the path specified below for statistics summary
    logger.info("Summarizing evaluation results...")
    for sample_id in tqdm(sample_ids):
        pos_filepath = str(
            Path(os.getenv("GROUP_RESPONSES_PATH"))
            / benchmark
            / args.model_type
            / args.prompt_strategy
            / f"group_responses_{sample_id}.csv"
        )
        neg_filepath = str(
            Path(os.getenv("GROUP_RESPONSES_PATH"))
            / benchmark
            / args.model_type
            / args.prompt_strategy
            / f"group_responses_{sample_id}_counter.csv"
        )

        pos_acc, neg_acc = calculate_group_accuracy(
            str(sample_id),
            pos_filepath,
            neg_filepath,
            benchmark,
        )
        pos_results.append(pos_acc)
        neg_results.append(neg_acc)

    group_eval_results = {
        "sample_ids": sample_ids,
        "group_acc": pos_results,
        "counter_group_acc": neg_results,
    }
    group_eval_results_df = pd.DataFrame(group_eval_results)
    save_dir = (
        Path(os.getenv("EVAL_HUMAN_ACCEPTED_DIR"))
        / benchmark
        / args.model_type
        / args.prompt_strategy
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"eval_results.csv"
    group_eval_results_df.to_csv(
        save_path,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Groups")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gemini-2.0-flash", "gpt-4o", "qwen-vl-max"],
        help="type of the model",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="seedbench_img",
        choices=SUPPORTED_BENCHMARKS,
        help="benchmark",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="direct_prompting",
        choices=["direct_prompting", "chain_of_thought", "guidance"],
        help="prompting strategy",
    )
    parser.add_argument(
        "--re-evaluate-group-responses",
        action="store_true",
        help="Whether to re-evaluate group responses if files exist",
        default=False,
    )
    args = parser.parse_args()
    main(args)
