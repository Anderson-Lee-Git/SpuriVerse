import os
from PIL import Image
import textwrap
import pandas as pd
import logging
from dotenv import load_dotenv

from pathlib import Path
import sys

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_preparation import (
    get_image_path_by_id,
    get_sample_by_image_id,
    get_benchmark_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_finetune_configuration(args):
    finetune_configuration = f"{args.data_type}_{args.layer}"
    if args.train_subsample:
        finetune_configuration = f"{finetune_configuration}_{args.train_subsample_size}"
    return finetune_configuration


def get_finetuning_save_path(args):
    """
    outputs/{model_type}/{data_type}_{train_subsample_size}/seed_{seed}
    """
    output_dir = os.getenv("FINETUNE_OUTPUT_DIR")
    output_dir = os.path.join(output_dir, args.model_type)
    finetune_configuration = get_finetune_configuration(args)
    output_dir = os.path.join(output_dir, finetune_configuration)
    save_path = os.path.join(output_dir, f"seed_{args.seed}")
    return save_path


def resize_image(image, image_size):
    if image_size is None:
        image_size = 512
    wpercent = image_size / image.size[0]
    hsize = int(image.size[1] * wpercent)
    image = image.resize((image_size, hsize), Image.Resampling.LANCZOS)
    return image


def get_image_paths_per_dir(directory):
    image_paths = []
    for f in os.listdir(directory):
        if not f.endswith(".png"):
            continue
        image_path = os.path.join(directory, f)
        image_paths.append(image_path)
    return image_paths


def convert_to_conversation(question, answer, image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    return {"messages": conversation}


def prepare_finetuning_conversations(df):
    """
    Prepare the conversations for the finetuning dataset where
    df should consist of sample ids and benchmarks
    """
    conversations = []
    for index, row in df.iterrows():
        sample = get_sample_by_image_id(row["sample_ids"], row["benchmark"])
        question = f"""
            Question: {sample['question']}
            Options:
            (A) {sample['A']}
            (B) {sample['B']}
            (C) {sample['C']}
            (D) {sample['D']}
            Please select the correct answer from the options above.
        """
        question = textwrap.dedent(question)
        image_path = get_image_path_by_id(row["sample_ids"], row["benchmark"])
        image = Image.open(image_path)
        answer_key = sample["answer"]
        answer = f"({answer_key}) {sample[answer_key]}"
        conversation = convert_to_conversation(question, answer, image)
        conversations.append(conversation)
    return conversations


def prepare_finetuning_spurious_group_conversations(df, num_samples=10):
    conversations = []
    for index, row in df.iterrows():
        sample_id = row["sample_ids"]
        sample = get_sample_by_image_id(sample_id, row["benchmark"])
        question = f"""
            Question: {sample['question']}
            Options:
            (A) {sample['A']}
            (B) {sample['B']}
            (C) {sample['C']}
            (D) {sample['D']}
            Please select the correct answer from the options above.
        """
        question = textwrap.dedent(question)
        answer_key = sample["answer"]
        answer = f"({answer_key}) {sample[answer_key]}"
        directory = os.path.join(
            os.getenv("GROUP_GENERATION_PROD_PATH"),
            row["benchmark"],
            f"{sample_id}",
        )
        image_paths = get_image_paths_per_dir(directory)
        image_paths = image_paths[:num_samples]
        for image_path in image_paths:
            image = Image.open(image_path)
            conversation = convert_to_conversation(question, answer, image)
            conversations.append(conversation)
    return conversations


def prepare_finetuning_non_spurious_conversations(
    full_df, train_df, val_df, seed, scale_factor=1
):
    """
    Scale the number of samples for each benchmark by a factor of scale_factor.
    """
    logger.info(f"Preparing non-spurious data...")
    logger.info(f"Full set size: {full_df.shape}")
    logger.info(f"Train set size: {train_df.shape}")
    logger.info(f"Val set size: {val_df.shape}")

    scale_factor = int(scale_factor)
    benchmark_counts = full_df["benchmark"].value_counts()

    naturalbench_meta = get_benchmark_metadata("naturalbench")
    aokvqa_meta = get_benchmark_metadata("aokvqa")
    seedbench_img_meta = get_benchmark_metadata("seedbench_img")
    seedbench2_meta = get_benchmark_metadata("seedbench2")

    naturalbench_sampled = naturalbench_meta.sample(
        n=benchmark_counts.get("naturalbench", 0) * scale_factor, random_state=seed
    )
    aokvqa_sampled = aokvqa_meta.sample(
        n=benchmark_counts.get("aokvqa", 0) * scale_factor, random_state=seed
    )
    seedbench_img_sampled = seedbench_img_meta.sample(
        n=benchmark_counts.get("seedbench_img", 0) * scale_factor, random_state=seed
    )
    seedbench2_sampled = seedbench2_meta.sample(
        n=benchmark_counts.get("seedbench2", 0) * scale_factor, random_state=seed
    )

    naturalbench_sampled["benchmark"] = "naturalbench"
    aokvqa_sampled["benchmark"] = "aokvqa"
    seedbench_img_sampled["benchmark"] = "seedbench_img"
    seedbench2_sampled["benchmark"] = "seedbench2"

    naturalbench_sampled["sample_ids"] = naturalbench_sampled["index"].astype(str)
    aokvqa_sampled["sample_ids"] = aokvqa_sampled["index"].astype(str)
    seedbench_img_sampled["sample_ids"] = seedbench_img_sampled["index"].astype(str)
    seedbench2_sampled["sample_ids"] = seedbench2_sampled["index"].astype(str)

    # re-distribute the samples to train and val sets based on individual benchmark counts
    train_benchmark_counts = train_df["benchmark"].value_counts()
    val_benchmark_counts = val_df["benchmark"].value_counts()
    # shuffle each sampled benchmark set
    naturalbench_sampled = naturalbench_sampled.sample(frac=1, random_state=seed)
    aokvqa_sampled = aokvqa_sampled.sample(frac=1, random_state=seed)
    seedbench_img_sampled = seedbench_img_sampled.sample(frac=1, random_state=seed)
    seedbench2_sampled = seedbench2_sampled.sample(frac=1, random_state=seed)
    # let train and val take a set of them based on the counts
    naturalbench_train = naturalbench_sampled.head(
        train_benchmark_counts.get("naturalbench", 0) * scale_factor
    )
    naturalbench_val = naturalbench_sampled.tail(
        val_benchmark_counts.get("naturalbench", 0) * scale_factor
    )
    aokvqa_train = aokvqa_sampled.head(
        train_benchmark_counts.get("aokvqa", 0) * scale_factor
    )
    aokvqa_val = aokvqa_sampled.tail(
        val_benchmark_counts.get("aokvqa", 0) * scale_factor
    )
    seedbench_img_train = seedbench_img_sampled.head(
        train_benchmark_counts.get("seedbench_img", 0) * scale_factor
    )
    seedbench_img_val = seedbench_img_sampled.tail(
        val_benchmark_counts.get("seedbench_img", 0) * scale_factor
    )
    seedbench2_train = seedbench2_sampled.head(
        train_benchmark_counts.get("seedbench2", 0) * scale_factor
    )
    seedbench2_val = seedbench2_sampled.tail(
        val_benchmark_counts.get("seedbench2", 0) * scale_factor
    )

    train_df_concat = pd.concat(
        [naturalbench_train, aokvqa_train, seedbench_img_train, seedbench2_train],
        ignore_index=True,
    )
    val_df_concat = pd.concat(
        [naturalbench_val, aokvqa_val, seedbench_img_val, seedbench2_val],
        ignore_index=True,
    )

    logger.info(f"Prepared non-spurious data for training set: {train_df_concat.shape}")
    logger.info(f"Prepared non-spurious data for validation set: {val_df_concat.shape}")

    return prepare_finetuning_conversations(
        train_df_concat
    ), prepare_finetuning_conversations(val_df_concat)


def sample_non_spurious_images(df, scale_factor=1, random_state=42):
    random_state = random_state * 2 + 1
    scale_factor = int(scale_factor)
    benchmark_counts = df["benchmark"].value_counts()

    naturalbench_meta = get_benchmark_metadata("naturalbench")
    aokvqa_meta = get_benchmark_metadata("aokvqa")
    seedbench_img_meta = get_benchmark_metadata("seedbench_img")
    seedbench2_meta = get_benchmark_metadata("seedbench2")

    naturalbench_sampled = naturalbench_meta.sample(
        n=benchmark_counts.get("naturalbench", 0) * scale_factor,
        random_state=random_state,
    )
    aokvqa_sampled = aokvqa_meta.sample(
        n=benchmark_counts.get("aokvqa", 0) * scale_factor, random_state=random_state
    )
    seedbench_img_sampled = seedbench_img_meta.sample(
        n=benchmark_counts.get("seedbench_img", 0) * scale_factor,
        random_state=random_state,
    )
    seedbench2_sampled = seedbench2_meta.sample(
        n=benchmark_counts.get("seedbench2", 0) * scale_factor,
        random_state=random_state,
    )

    naturalbench_sampled["benchmark"] = "naturalbench"
    aokvqa_sampled["benchmark"] = "aokvqa"
    seedbench_img_sampled["benchmark"] = "seedbench_img"
    seedbench2_sampled["benchmark"] = "seedbench2"

    naturalbench_sampled["sample_ids"] = naturalbench_sampled["index"].astype(str)
    aokvqa_sampled["sample_ids"] = aokvqa_sampled["index"].astype(str)
    seedbench_img_sampled["sample_ids"] = seedbench_img_sampled["index"].astype(str)
    seedbench2_sampled["sample_ids"] = seedbench2_sampled["index"].astype(str)

    return pd.concat(
        [
            naturalbench_sampled,
            aokvqa_sampled,
            seedbench_img_sampled,
            seedbench2_sampled,
        ],
        ignore_index=True,
    )


def convert_to_message(question):
    message = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }
    ]
    return message
