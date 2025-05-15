import argparse
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from utils.data_preparation import encode_image, get_image_path_by_id
from utils.constants import SUPPORTED_BENCHMARKS

load_dotenv()


def main(args):
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    if args.benchmark == "aokvqa":
        md_path = os.getenv("AOKVQA_METADATA_PATH")
    elif args.benchmark == "seedbench_img":
        md_path = os.getenv("SEEDBENCH_IMG_METADATA_PATH")
    elif args.benchmark == "seedbench2":
        md_path = os.getenv("SEEDBENCH2_METADATA_PATH")
    elif args.benchmark == "naturalbench":
        md_path = os.getenv("NATURALBENCH_METADATA_PATH")
    df = pd.read_csv(md_path)
    assert (
        "isError" in df.columns
    ), "isError column must be present in the metadata file"
    error_df = df.loc[df["isError"]]
    spu_attr = []
    print(f"Number of original samples: {df.shape[0]}")
    print(f"Number of error samples: {error_df.shape[0]}")
    pbar = tqdm(total=error_df.shape[0])
    for index, row in error_df.iterrows():
        image_id = row["index"]
        base64_image = encode_image(get_image_path_by_id(image_id, args.benchmark))
        context = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant to determine if a model's error is caused primarily by spurious correlations, patterns that can often be used to predict the target, but are not actually causal. You need to begin your response with Yes or No.",
                    }
                ],
            },
        ]
        context.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given this image, a Large Multi-modal Model was asked, {row['question']}, and given the choices: A) {row['A']} B) {row['B']} C) {row['C']} D) {row['D']}.  The model chose {row['parsed']} and the correct answer is {row['answer']}. Is the error primarily caused by spurious correlation?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=context,
                max_tokens=500,
                temperature=0,
                top_p=1,
            )
            response_content = response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
        spu_attr.append(response_content)
        pbar.update(1)
    error_df["spuriousAttr"] = spu_attr
    error_df["isSpurious"] = error_df["spuriousAttr"].str[0] == "Y"
    # Update the metadata file
    df = pd.merge(
        df,
        error_df,
        how="left",
        on="index",
        suffix=(None, "_error_set"),
    )
    # remove duplicate columns
    df = df.loc[:, ~df.columns.str.endswith("_error_set")]
    df["isError"] = df["isError"].fillna(False)
    df["isSpurious"] = df["isSpurious"].fillna(False)
    # Reset index if needed
    if args.write_to_metadata:
        df.to_csv(md_path, index=False)
    # save filtered error df
    filtered_df = error_df.loc[error_df["isSpurious"]]
    filtered_df.to_csv(
        Path(os.getenv("VLM_ACCEPTED_DIR")) / f"{args.benchmark}_vlm_accepted.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=SUPPORTED_BENCHMARKS,
        help="Benchmark to filter",
    )
    parser.add_argument(
        "--write-to-metadata",
        action="store_true",
        help="Write the filtered results to the metadata file",
    )
    args = parser.parse_args()
    main(args)
