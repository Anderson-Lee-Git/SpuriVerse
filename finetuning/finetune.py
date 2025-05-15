import argparse
import pandas as pd
import os
import shutil
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.finetuning_utils import (
    get_finetuning_save_path,
    get_finetune_configuration,
    prepare_finetuning_conversations,
    prepare_finetuning_spurious_group_conversations,
    prepare_finetuning_non_spurious_conversations,
    sample_non_spurious_images,
)
from utils.evaluation import (
    compute_acc,
)
from finetuning.evaluation import (
    evaluate_benchmark_samples,
    evaluate_group,
    wandb_log,
)
from sklearn.model_selection import train_test_split
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from dotenv import load_dotenv
import logging

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def evaluate(
    args,
    model,
    tokenizer,
    test_df,
):
    """
    Test suite contains three parts:
    1. Anchor set (test_df)
    2. Spurious group set (test_df's derivative)
    3. Non-spurious set (sample images from benchmarks per test_df's distribution)
    """
    FastVisionModel.for_inference(model)
    # anchor part
    anchor_eval_results_df = evaluate_benchmark_samples(model, tokenizer, test_df)
    # spurious group part
    group_eval_results_df = evaluate_group(model, tokenizer, test_df)
    # non-spurious part
    non_spurious_df = sample_non_spurious_images(
        test_df, scale_factor=10
    )
    non_spurious_eval_results_df = evaluate_benchmark_samples(
        model, tokenizer, non_spurious_df
    )
    anchor_acc = compute_acc(anchor_eval_results_df)
    spurious_group_acc = compute_acc(group_eval_results_df)
    non_spurious_acc = compute_acc(non_spurious_eval_results_df)
    if args.use_wandb:
        wandb_log(
            args,
            anchor_acc,
            spurious_group_acc,
            non_spurious_acc,
        )
    logger.info(
        f"Evaluation Result for {args.model_type} (finetuned with {get_finetune_configuration(args)})"
    )
    logger.info(f"Anchor set accuracy: {anchor_acc}")
    logger.info(f"Spurious group set accuracy: {spurious_group_acc}")
    logger.info(f"Non-spurious set accuracy: {non_spurious_acc}")
    return anchor_acc, spurious_group_acc, non_spurious_acc


def main(args):
    anchor_set_path = (
        Path(os.getenv("EVAL_HUMAN_ACCEPTED_DIR")) / "reference_anchor_set.csv"
    )
    df = pd.read_csv(anchor_set_path)
    seed = args.seed

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    # 0.125 x 0.8 = 0.1 of the whole set
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=seed)
    complementary_train_df = None
    if args.train_subsample:
        """
        Subsample the spurious anchor set; the complimentary is the non-spurious samples following
        the same distribution of the non-sampled ones in the rest of the spurious anchor set.
        """
        # reconcat the train and val sets
        train_df = pd.concat([train_df, val_df])
        if args.train_subsample_size is None:
            raise ValueError(
                "train_subsample_size must be specified if train_subsample is True"
            )
        if args.train_subsample_size > train_df.shape[0]:
            raise ValueError(
                "train_subsample_size must be less than the number of training samples"
            )
        if args.data_type != "anchor" and args.data_type != "group":
            raise ValueError(
                "train_subsample is only supported for anchor and group set"
            )
        # Shuffle and subsample the training data
        shuffled_train_df = train_df.sample(frac=1, random_state=seed)
        # Take first n samples as new training set
        train_df = shuffled_train_df.iloc[: args.train_subsample_size]
        # Get the complementary set as the remaining samples
        complementary_train_df = shuffled_train_df.iloc[args.train_subsample_size :]
        val_df = pd.DataFrame(columns=train_df.columns)

    logger.info(f"Dataset sizes from dataframe:")
    logger.info(f"Train set size: {train_df.shape}")
    logger.info(f"Val set size: {val_df.shape}")
    logger.info(f"Test set size: {test_df.shape}")

    if args.data_type == "anchor" and args.train_subsample:
        logger.info(f"Complementary training set size: {complementary_train_df.shape}")
        assert complementary_train_df is not None
        train_data = prepare_finetuning_conversations(train_df)
        complementary_train_data, _ = prepare_finetuning_non_spurious_conversations(
            full_df=df,
            train_df=complementary_train_df,
            val_df=val_df,  # should be empty
            seed=seed,
            scale_factor=1,
            error_set=False,
        )
        train_data = train_data + complementary_train_data
        train_data, val_data = train_test_split(
            train_data, test_size=0.125, random_state=seed
        )
    elif args.data_type == "group" and args.train_subsample:
        logger.info(f"Complementary training set size: {complementary_train_df.shape}")
        assert complementary_train_df is not None
        train_data = prepare_finetuning_spurious_group_conversations(
            train_df, num_samples=10
        )
        complementary_train_data, _ = prepare_finetuning_non_spurious_conversations(
            full_df=df,
            train_df=complementary_train_df,
            val_df=val_df,  # should be empty
            seed=seed,
            scale_factor=args.scale_factor,
            error_set=False,
        )
        train_data = train_data + complementary_train_data
        train_data, val_data = train_test_split(
            train_data, test_size=0.125, random_state=seed
        )
    elif args.data_type == "anchor":
        train_data = prepare_finetuning_conversations(train_df)
        val_data = prepare_finetuning_conversations(val_df)
    elif args.data_type == "group":
        train_data = prepare_finetuning_spurious_group_conversations(
            train_df, num_samples=10
        )
        val_data = prepare_finetuning_spurious_group_conversations(
            val_df, num_samples=10
        )
    elif args.data_type == "non_spurious":
        train_data, val_data = prepare_finetuning_non_spurious_conversations(
            full_df=df,
            train_df=train_df,
            val_df=val_df,
            seed=seed,
            scale_factor=args.scale_factor,
        )
    elif args.data_type == "mixed":
        group_train_data = prepare_finetuning_spurious_group_conversations(
            train_df, num_samples=10
        )
        group_val_data = prepare_finetuning_spurious_group_conversations(
            val_df, num_samples=10
        )
        non_spurious_train_data, non_spurious_val_data = (
            prepare_finetuning_non_spurious_conversations(
                full_df=df,
                train_df=train_df,
                val_df=val_df,
                seed=seed,
                scale_factor=args.scale_factor,
            )
        )
        train_data = group_train_data + non_spurious_train_data
        val_data = group_val_data + non_spurious_val_data
    else:
        raise ValueError("Invalid data_type option")
    logger.info(f"Dataset sizes from prepared data:")
    logger.info(f"Train set size: {len(train_data)}")
    logger.info(f"Val set size: {len(val_data)}")
    save_path = get_finetuning_save_path(args)
    logger.info(f"Will save finetuned model to: {save_path}")
    if os.path.exists(save_path):
        # move the existing folder to a backup folder
        backup_path = save_path.replace("outputs", "backup")
        logger.info(f"Moving existing folder to backup: {backup_path}")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(save_path, "/".join(backup_path.split("/")[:-1]))
    os.makedirs(save_path, exist_ok=True)

    model_name = None
    if args.model_type == "llama":
        model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
    elif args.model_type == "qwen":
        model_name = "unsloth/Qwen2.5-VL-7B-Instruct"
    else:
        raise ValueError("Invalid model_type option")
    assert model_name is not None
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        # Use 4bit to reduce memory use. False for 16bit LoRA.
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        cache_dir=os.getenv("MODEL_CACHE_DIR"),
    )

    finetune_language = False
    finetune_vision = False
    if args.layer == "language":
        finetune_language = True
    elif args.layer == "vision":
        finetune_vision = True
    elif args.layer == "both":
        finetune_language = True
        finetune_vision = True

    model = FastVisionModel.get_peft_model(
        model,
        # False if not finetuning vision layers
        finetune_vision_layers=finetune_vision,
        # False if not finetuning language layers
        finetune_language_layers=finetune_language,
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    FastVisionModel.for_training(model)  # Enable for training!

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_data,
        eval_dataset=val_data,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.num_epochs,  # Set this instead of max_steps for full training runs
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=save_path,
            report_to="none",  # For Weights and Biases
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            save_strategy="best",
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # For loss, lower is better
            save_total_limit=2,
        ),
    )
    trainer.train()
    # Save the best model
    # the best model is loaded at the end
    best_checkpoint_save_path = os.path.join(save_path, "best-checkpoint")
    trainer.save_model(best_checkpoint_save_path)

    evaluate(args, trainer.model, tokenizer, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="number of training epochs"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="type of the model",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="anchor",
        choices=[
            "anchor",
            "group",
            "non_spurious",
            "mixed",
        ],
    )
    parser.add_argument(
        "--layer", type=str, default="both", help="language, vision, or both"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for train-test split, must aligned with evaluation split",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1, help="scale factor for non_spurious"
    )
    parser.add_argument(
        "--train-subsample",
        action="store_true",
        help="subsample training data",
        default=False,
    )
    parser.add_argument(
        "--train-subsample-size",
        type=int,
        default=None,
        help="subsample size for training",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="use wandb", default=False
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="debug-eval",
        help="wandb project name",
    )
    args = parser.parse_args()
    main(args)
