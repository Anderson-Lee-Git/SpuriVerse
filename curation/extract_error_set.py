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
    get_benchmark_id_set,
    parse_llava_data,
    parse_llava_15_data,
)
from utils.constants import SUPPORTED_BENCHMARKS
from utils.evaluation import compute_acc
from curation.engine import get_prediction, VLMInterface

load_dotenv()


def main(args):
    """
    Evaluates model on full benchmark to extract the error set
    """
    sample_ids_to_evaluate = list(get_benchmark_id_set(args.benchmark))

    predictions = []
    answers = []
    model = VLMInterface(model_type=args.model_type)

    for sample_id in tqdm(sample_ids_to_evaluate):
        pred = get_prediction(
            image_id=str(sample_id),
            model=model,
            benchmark=args.benchmark,
            prompt_strategy=args.prompt_strategy,
        )
        sample = get_sample_by_image_id(str(sample_id), args.benchmark)
        predictions.append(pred)
        answers.append(sample["answer"])

    if args.model_type == "llava-1.6":
        parsed = [parse_llava_data(pred) for pred in predictions]
    elif args.model_type == "llava-1.5":
        parsed = [parse_llava_15_data(pred) for pred in predictions]
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
        parsed = [parse_data_w_gpt(pred, client) for pred in predictions]
    else:
        parsed = [parse_data(pred) for pred in predictions]

    eval_results = {
        "index": sample_ids_to_evaluate,
        "prediction": predictions,
        "parsed": parsed,
        "answer": answers,
    }

    eval_results_df = pd.DataFrame(eval_results)
    # Save DataFrame to a CSV file
    eval_results_df.to_csv(
        Path(os.getenv("FULL_EVAL_RESULTS_DIR"))
        / args.model_type
        / f"{args.benchmark}_{args.prompt_strategy}_eval.csv",
        index=False,
    )

    acc = compute_acc(eval_results_df)
    print(f"Accuracy: {acc}")

    # OVERRIDING the metadata file with predictions and parsed predictions
    if args.write_to_metadata:
        if args.benchmark == "seedbench_img":
            md_path = os.getenv("SEEDBENCH_IMG_METADATA_PATH")
        elif args.benchmark == "seedbench2":
            md_path = os.getenv("SEEDBENCH2_METADATA_PATH")
        elif args.benchmark == "aokvqa":
            md_path = os.getenv("AOKVQA_METADATA_PATH")
        elif args.benchmark == "naturalbench":
            md_path = os.getenv("NATURALBENCH_METADATA_PATH")
        md = pd.read_csv(md_path)
        # Has to be merge on "index", otherwise the index might be messed up
        md["index"] = md["index"].astype(str)
        eval_results_df["index"] = eval_results_df["index"].astype(str)

        md = pd.merge(
            md,
            eval_results_df[["index", "prediction", "parsed"]],
            on="index",
            how="inner",
        )
        md["isError"] = md["prediction"] != md["answer"]
        md.to_csv(md_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate seed spurious correlation examples"
    )
    parser.add_argument(
        "--model-type", type=str, default="gpt-4o", help="type of the model"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=SUPPORTED_BENCHMARKS,
        help=f"benchmark to evaluate in {SUPPORTED_BENCHMARKS}",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="direct_prompting",
        help="prompting strategy",
    )
    parser.add_argument(
        "--write-to-metadata",
        action="store_true",
        help="write the results to the metadata file",
    )
    args = parser.parse_args()
    main(args)
