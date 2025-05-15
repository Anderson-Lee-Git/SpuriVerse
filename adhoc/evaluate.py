import os
import argparse

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_preparation import (
    parse_data,
    parse_data_w_gpt,
    get_sample_by_image_id,
    parse_llava_data,
    parse_llava_15_data,
    convert_to_str,
)
from utils.finetuning_utils import sample_non_spurious_images
from utils.constants import SUPPORTED_MODELS
from utils.evaluation import compute_acc
from curation.engine import get_prediction, VLMInterface
from curation.pipeline import Pipeline

load_dotenv()


def parse_predictions(predictions, args):
    if args.model_type == "llava-1.6":
        parsed = [parse_llava_data(pred) for pred in tqdm(predictions)]
    elif args.model_type == "llava-1.5":
        parsed = [parse_llava_15_data(pred) for pred in tqdm(predictions)]
    elif (
        args.model_type == "qwen-2.5-7b"
        or args.model_type == "qwen-2.5-32b"
        or args.model_type == "qwen-2.5-72b"
        or args.model_type == "llama-3.2-11b"
        or args.model_type == "llama-3.2-90b"
        or args.model_type == "o4-mini"
        or args.model_type == "o3"
    ):
        client = OpenAI(api_key=os.environ["OPENAI_KEY"])
        parsed = [parse_data_w_gpt(pred, client) for pred in tqdm(predictions)]
    else:
        parsed = [parse_data(pred) for pred in tqdm(predictions)]
    return parsed


def main(args):
    """
    Evaluates model on full benchmark to extract the error set
    """
    anchor_set_path = (
        Path(os.getenv("EVAL_HUMAN_ACCEPTED_DIR")) / "reference_anchor_set.csv"
    )
    anchor_set_df = pd.read_csv(anchor_set_path)
    anchor_predictions = []
    spurious_group_predictions = []
    non_spurious_predictions = []
    anchor_answers = []
    spurious_group_answers = []
    non_spurious_answers = []
    anchor_parsed = None
    spurious_group_parsed = None
    non_spurious_parsed = None
    model = VLMInterface(model_type=args.model_type)
    # anchor set
    if args.anchor:
        pbar = tqdm(total=len(anchor_set_df))
        for index, row in anchor_set_df.iterrows():
            if type(row["sample_ids"]) == str:
                sample_id = row["sample_ids"]
            else:
                sample_id = str(int(row["sample_ids"]))
            pred = get_prediction(
                image_id=sample_id,
                model=model,
                benchmark=row["benchmark"],
                prompt_strategy=args.prompt_strategy,
            )
            sample = get_sample_by_image_id(str(sample_id), row["benchmark"])
            anchor_predictions.append(pred)
            anchor_answers.append(sample["answer"])
            pbar.update(1)
        pbar.close()
        anchor_parsed = parse_predictions(anchor_predictions, args)
        eval_results = {
            "index": anchor_set_df["sample_ids"].apply(convert_to_str),
            "benchmark": anchor_set_df["benchmark"],
            "prediction": anchor_predictions,
            "parsed": anchor_parsed,
            "answer": anchor_answers,
        }
        eval_results_df = pd.DataFrame(eval_results)
        save_dir = (
            Path(os.getenv("ADHOC_EVAL_RESULTS_DIR"))
            / args.model_type
            / f"{args.prompt_strategy}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save DataFrame to a CSV file
        eval_results_df.to_csv(
            save_dir / "anchor_eval_results.csv",
            index=False,
        )
        acc = compute_acc(eval_results_df)
        print(f"Anchor set accuracy: {acc}")
    # spurious group set
    if args.spurious_group:
        spurious_group_sample_ids = []
        spurious_group_benchmarks = []
        pbar = tqdm(total=len(anchor_set_df))
        for index, row in anchor_set_df.iterrows():
            if type(row["sample_ids"]) == str:
                sample_id = row["sample_ids"]
            else:
                sample_id = str(int(row["sample_ids"]))
            benchmark = row["benchmark"]
            image_directory = (
                Path(os.getenv("GROUP_GENERATION_PROD_PATH")) / benchmark / sample_id
            )
            sample = get_sample_by_image_id(str(sample_id), benchmark)
            eval_results = Pipeline.evaluate(
                image_id=sample_id,
                directory=image_directory,
                model=model,
                model_type=args.model_type,
                benchmark=benchmark,
                prompt_strategy=args.prompt_strategy,
            )
            spurious_group_predictions.extend(eval_results["response"].to_list())
            spurious_group_answers.extend(sample["answer"] * len(eval_results))
            spurious_group_sample_ids.extend([sample_id] * len(eval_results))
            spurious_group_benchmarks.extend([benchmark] * len(eval_results))
            pbar.update(1)
        pbar.close()
        spurious_group_parsed = parse_predictions(spurious_group_predictions, args)
        eval_results = {
            "index": spurious_group_sample_ids,
            "benchmark": spurious_group_benchmarks,
            "prediction": spurious_group_predictions,
            "parsed": spurious_group_parsed,
            "answer": spurious_group_answers,
        }
        eval_results_df = pd.DataFrame(eval_results)
        save_dir = (
            Path(os.getenv("ADHOC_EVAL_RESULTS_DIR"))
            / args.model_type
            / f"{args.prompt_strategy}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save DataFrame to a CSV file
        eval_results_df.to_csv(
            save_dir / "spurious_group_eval_results.csv",
            index=False,
        )
        acc = compute_acc(eval_results_df)
        print(f"Spurious group set accuracy: {acc}")
    # non-spurious set
    if args.non_spurious:
        non_spurious_df = sample_non_spurious_images(anchor_set_df, scale_factor=10)
        pbar = tqdm(total=len(non_spurious_df))
        for index, row in non_spurious_df.iterrows():
            if type(row["sample_ids"]) == str:
                sample_id = row["sample_ids"]
            else:
                sample_id = str(int(row["sample_ids"]))
            pred = get_prediction(
                image_id=sample_id,
                model=model,
                benchmark=row["benchmark"],
                prompt_strategy=args.prompt_strategy,
            )
            sample = get_sample_by_image_id(str(sample_id), row["benchmark"])
            non_spurious_predictions.append(pred)
            non_spurious_answers.append(sample["answer"])
            pbar.update(1)
        pbar.close()
        non_spurious_parsed = parse_predictions(non_spurious_predictions, args)
        eval_results = {
            "index": non_spurious_df["sample_ids"].apply(convert_to_str),
            "benchmark": non_spurious_df["benchmark"],
            "prediction": non_spurious_predictions,
            "parsed": non_spurious_parsed,
            "answer": non_spurious_answers,
        }
        eval_results_df = pd.DataFrame(eval_results)
        # Save DataFrame to a CSV file
        eval_results_df.to_csv(
            save_dir / "non_spurious_eval_results.csv",
            index=False,
        )
        acc = compute_acc(eval_results_df)
        print(f"Non-spurious set accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate seed spurious correlation examples"
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
        help="evaluate anchor set",
    )
    parser.add_argument(
        "--spurious-group",
        action="store_true",
        help="evaluate spurious group set",
    )
    parser.add_argument(
        "--non-spurious",
        action="store_true",
        help="evaluate non-spurious set; note that you need the full dataset downloaded to use this",
    )
    args = parser.parse_args()
    main(args)
