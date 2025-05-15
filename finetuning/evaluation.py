from openai import OpenAI
import pandas as pd
import os
import textwrap
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.finetuning_utils import (
    resize_image,
    convert_to_message,
    get_image_paths_per_dir,
    get_finetune_configuration,
)
from utils.data_preparation import (
    get_sample_by_image_id,
    get_image_path_by_id,
    parse_data_w_gpt,
)


def evaluate_benchmark_samples(
    model,
    tokenizer,
    test_df,
):
    predictions = []
    answers = []
    parsed = []
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
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
        # resize image the same way as UnslothVisionDataCollator
        image_path = get_image_path_by_id(row["sample_ids"], row["benchmark"])
        image = Image.open(image_path)
        # resize image the same way as UnslothVisionDataCollator
        if hasattr(model, "vision_config") and hasattr(
            model.vision_config, "image_size"
        ):
            image = resize_image(image, image_size=model.vision_config.image_size)
        else:
            image = resize_image(image, image_size=None)
        answer_key = sample["answer"]
        instruction = convert_to_message(question)
        input_text = tokenizer.apply_chat_template(
            instruction, add_generation_prompt=True
        )
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        response = model.generate(
            **inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1
        )
        decoded = tokenizer.decode(response[0], skip_special_tokens=True)
        predictions.append(decoded)
        answers.append(answer_key)
        parsed_response = parse_data_w_gpt(decoded, client)
        parsed.append(parsed_response)

    eval_results = {
        "sample_id": test_df["sample_ids"].apply(str).values,
        "prediction": predictions,
        "parsed": parsed,
        "answer": answers,
    }
    eval_results_df = pd.DataFrame(eval_results)
    return eval_results_df


def evaluate_group(model, tokenizer, test_df):
    sample_paths = []
    predictions = []
    answers = []
    parsed = []
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
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
        instruction = convert_to_message(question)
        answer_key = sample["answer"]
        directory = os.path.join(
            os.getenv("GROUP_GENERATION_PROD_PATH"),
            row["benchmark"],
            sample_id,
        )
        image_paths = get_image_paths_per_dir(directory)
        for image_path in image_paths:
            image = Image.open(image_path)
            # resize image the same way as UnslothVisionDataCollator
            if hasattr(model, "vision_config") and hasattr(
                model.vision_config, "image_size"
            ):
                image = resize_image(image, image_size=model.vision_config.image_size)
            else:
                image = resize_image(image, image_size=None)
            input_text = tokenizer.apply_chat_template(
                instruction, add_generation_prompt=True
            )
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            response = model.generate(
                **inputs, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1
            )
            decoded = tokenizer.decode(response[0], skip_special_tokens=True)
            predictions.append(decoded)
            answers.append(answer_key)
            parsed_response = parse_data_w_gpt(decoded, client)
            parsed.append(parsed_response)
            sample_paths.append(image_path)

    eval_results = {
        "sample_path": sample_paths,
        "prediction": predictions,
        "parsed": parsed,
        "answer": answers,
    }
    eval_results_df = pd.DataFrame(eval_results)
    return eval_results_df


def wandb_log(
    args,
    anchor_acc,
    spurious_group_acc,
    non_spurious_acc,
):
    finetune_configuration = get_finetune_configuration(args)
    seed = args.seed
    run = wandb.init(
        entity="vl-spurious-corr",
        project=args.wandb_project_name,
        job_type="debug",
        name=f"{args.model_type}_{finetune_configuration}_seed_{seed}",
        config={
            "model_type": args.model_type,
            "finetune_setup": finetune_configuration,
            "random_seed": seed,
        },
        reinit=True,  # Allow multiple wandb.init calls in the same process
    )
    run.log(
        data={
            "anchor_acc": anchor_acc,
            "spurious_group_acc": spurious_group_acc,
            "non_spurious_acc": non_spurious_acc,
        }
    )
