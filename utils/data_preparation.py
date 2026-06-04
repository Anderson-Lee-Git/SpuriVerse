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
    """
    Robustly parse a multiple-choice answer from free-form model output.
    Priority:
      1) A line that is exactly one of: (A), (B), (C), (D) or A/B/C/D
      2) Phrases like 'final answer: (B)' or 'answer: C'
      3) Parenthesized choice anywhere, prefer the last occurrence
    Avoid naive substring checks to prevent defaulting to 'A'.
    Return 'E' if nothing reliable is found.
    """
    import re

    if not isinstance(data, str):
        return "E"

    text = data.strip()
    if not text:
        return "E"

    # 1) Check line-by-line from bottom: exact single-letter choice
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        # Exact (X) or X
        m = re.fullmatch(r"\(?([A-E])\)?", ln, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2) Look for explicit 'final answer' or 'answer' cues
    m = re.search(r"final\s*answer\s*[:\-]?\s*\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) Any parenthesized choice; prefer the last one in the text
    candidates = re.findall(r"\(([A-E])\)", text, flags=re.IGNORECASE)
    if candidates:
        return candidates[-1].upper()

    # 4) Fallback: look for standalone letters surrounded by non-letters (less likely to collide)
    m = re.search(r"\b([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

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
    """
    LLaVA-1.6 outputs can contain multiple [/INST] markers due to the chat template.
    Extract the content strictly after the LAST [/INST] and parse that.
    """
    import re

    if not isinstance(data, str) or not data.strip():
        return "E"

    # Split on [/INST] and take the segment after the last occurrence
    parts = re.split(r"\[/INST\]", data, flags=re.DOTALL)
    if len(parts) >= 2:
        ret = parts[-1].strip()
    else:
        # Fallback to previous behavior (first match) if no split worked
        m = re.search(r"\[/INST\](.*)", data, re.DOTALL)
        ret = m.group(1).strip() if m else "(E)"

    # Sometimes responses may still include role prefixes; trim obvious ones
    # e.g., "ASSISTANT:" or similar headers
    ret = re.sub(r"^(ASSISTANT:|Assistant:|assistant:)\s*", "", ret)

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
