import os
import re
import base64
import pandas as pd
import textwrap
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import SUPPORTED_BENCHMARKS

load_dotenv()

logger = logging.getLogger(__name__)


def convert_to_str(value):
    if type(value) != str:
        return str(int(value))
    return value


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return None


def get_image_path_by_id(image_id, benchmark):
    assert benchmark in SUPPORTED_BENCHMARKS
    if benchmark == "aokvqa":
        coco_dir = os.path.join(
            os.getenv("AOKVQA_ROOT"), "datasets/coco/"
        )  # Directory to datasets/coco in aokvqa
        image_id = int(image_id)
        image_path = get_coco_path("train", image_id, coco_dir)
    elif benchmark == "seedbench_img":
        image_path = os.path.join(
            os.getenv("LMU_DATA_ROOT"), f"images/SEEDBench_IMG/{image_id}.jpg"
        )
    elif benchmark == "seedbench2":
        image_path = os.path.join(
            os.getenv("LMU_DATA_ROOT"), f"images/SEEDBench2/{image_id}.jpg"
        )
    elif benchmark == "naturalbench":
        image_path = os.path.join(
            os.getenv("LMU_DATA_ROOT"), f"images/NaturalBenchDataset/{image_id}.jpg"
        )

    return image_path


def get_benchmark_metadata(benchmark):
    assert benchmark in SUPPORTED_BENCHMARKS
    if benchmark == "aokvqa":
        return pd.read_csv(os.getenv("AOKVQA_METADATA_PATH"))
    elif benchmark == "seedbench_img":
        return pd.read_csv(os.getenv("SEEDBENCH_IMG_METADATA_PATH"))
    elif benchmark == "seedbench2":
        return pd.read_csv(os.getenv("SEEDBENCH2_METADATA_PATH"))
    elif benchmark == "naturalbench":
        return pd.read_csv(os.getenv("NATURALBENCH_METADATA_PATH"))


def get_sample_by_image_id(image_id: str, benchmark: str):
    """
    This only contains the samples from the error set (in the metadata)
    """
    assert isinstance(image_id, str), "image_id must be a string"
    assert benchmark in SUPPORTED_BENCHMARKS
    df = get_benchmark_metadata(benchmark)
    if not pd.api.types.is_string_dtype(df["index"]):
        df["index"] = df["index"].astype(str)
    rows = df.loc[df["index"] == image_id]
    return rows.iloc[0]


def get_benchmark_id_set(benchmark):
    assert benchmark in SUPPORTED_BENCHMARKS
    df = get_benchmark_metadata(benchmark)
    if not pd.api.types.is_string_dtype(df["index"]):
        df["index"] = df["index"].astype(str)
    return set(df["index"].values)


def get_is_filtered_benchmark_id_set(benchmark):
    assert benchmark in SUPPORTED_BENCHMARKS
    df = get_benchmark_metadata(benchmark)
    if not pd.api.types.is_string_dtype(df["index"]):
        df["index"] = df["index"].astype(str)
    id_set = set(df.loc[df["isSpurious"]]["index"].values)
    return id_set


def parse_data(data):
    if "A)" in data:
        return "A"
    elif "B)" in data:
        return "B"
    elif "C)" in data:
        return "C"
    elif "D)" in data:
        return "D"
    elif "E)" in data:
        return "E"
    elif "A." in data:
        return "A"
    elif "B." in data:
        return "B"
    elif "C." in data:
        return "C"
    elif "D." in data:
        return "D"
    elif "A" in data:
        return "A"
    elif "B" in data:
        return "B"
    elif "C" in data:
        return "C"
    elif "D" in data:
        return "D"
    elif "E" in data:
        return "E"


def parse_data_w_gpt(data, client):
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": textwrap.dedent(
                    """
                    You are given a multiple choice question and a response. You need to parse the response as (A), (B), (C), or (D). If none of the above, then return (E). You will not use any fullstops or punctuation. You will not explain your answers or write words before or after the answers. Only the answer itself will you respond with.
                """
                ),
            }
        ],
    }

    context = [system_prompt]
    context.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": data,
                },
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context,
            max_tokens=100,
            temperature=0,
            top_p=1,
        )
        ret = response.choices[0].message.content
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        ret = "(E)"

    return parse_data(ret)


def parse_llava_data(data):

    # Extract the text after [/INST]
    result = re.search(r"\[/INST\](.*)", data, re.DOTALL)

    # Output the extracted text
    if result:
        ret = result.group(1).strip()
    else:
        print("No text found after [/INST].")
        ret = "(E)"

    return parse_data(ret)


def parse_llava_15_data(data):

    # Regular expression to extract the content after "ASSISTANT:"
    content = data.split("ASSISTANT:")[-1].strip()

    if content:
        ret = content  # Extract the selected option
    else:
        print("No answer found after 'ASSISTANT:'")
        ret = "(E)"

    return parse_data(ret)


def idx2letter(idx):
    return chr(ord("A") + idx)


def letter2idx(letter):
    if letter == "A":
        return 0
    elif letter == "B":
        return 1
    elif letter == "C":
        return 2
    elif letter == "D":
        return 3
    else:
        raise ValueError("Invalid input detected")
