"""
This file is largely borrowed from Open-r1 (https://github.com/huggingface/open-r1)
"""

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
    entropy_based_reward,
    confidence_based_reward
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from peft import LoraConfig

logger = logging.getLogger(__name__)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------------------------------------
# Utility: token-length histogram for the reference solutions
# ------------------------------------------------------------
from collections import Counter
import math

def report_answer_length_ranges(dataset, answer_key: str, tokenizer, buckets=None):
    """
    Tokenises every item under `answer_key` and prints a histogram over length buckets.

    Parameters
    ----------
    dataset      :   HF DatasetDict or Dataset
    answer_key   :   column that holds the ground-truth answer / solution text
    tokenizer    :   Hugging-Face tokenizer (already initialised)
    buckets      :   iterable of upper bounds; e.g. (16, 32, 64, 128)
                     Everything above the last bound goes into an '>{last}' bin.

    Example
    -------
    buckets = (16, 32, 64, 128)  →   0-16, 17-32, 33-64, 65-128, >128
    """
    if buckets is None:
        buckets = (16, 32, 64, 128)          # sensible defaults

    # build bucket edges and labels once
    bounds  = list(buckets)
    labels  = [f"0-{bounds[0]}"]
    labels += [f"{bounds[i-1]+1}-{b}" for i, b in enumerate(bounds[1:], start=1)]
    labels += [f">{bounds[-1]}"]

    counter = Counter({lab: 0 for lab in labels})

    # iterate through ALL splits that contain `answer_key`
    for split in dataset:
        for ans in dataset[split][answer_key]:
            # Tokenise without special tokens
            token_ids = tokenizer.encode(ans, add_special_tokens=False)
            L = len(token_ids)

            # place length in the right bucket
            placed = False
            for upper, lab in zip(bounds, labels):
                if L <= upper:
                    counter[lab] += 1
                    placed = True
                    break
            if not placed:
                counter[labels[-1]] += 1     # overflow bucket

    # pretty print
    total = sum(counter.values())
    print("\n=== Answer length histogram ===")
    for lab in labels:
        n = counter[lab]
        pct = 100.0 * n / total
        bar = "█" * math.ceil(pct / 2)       # quick text bar-chart
        print(f"{lab:>10}: {n:>6} ({pct:5.1f}%) {bar}")
    print(f"Total answers: {total}\n")

def get_dataset(dataset_name, config):
    # Load the dataset based on the configuration
    dataset = load_dataset(dataset_name, name=config)
    
    # Define the possible column names to standardize
    problem_names = ["Problem", "question", "Question"]
    answer_names = ["answer", "Answer"]
    for split in dataset:
        # Rename problem-related columns to "problem"
        if "problem" not in dataset[split].column_names:
            for name in problem_names:
                if name in dataset[split].column_names:
                    dataset[split] = dataset[split].rename_column(name, "problem")
                    break
        
        # Rename answer-related columns to "answer"
        if "answer" not in dataset[split].column_names:
            for name in answer_names:
                if name in dataset[split].column_names:
                    dataset[split] = dataset[split].rename_column(name, "answer")
                    break
    return dataset


# Define script arguments for GRPO training
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

# Main function to run the training
def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ### Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log process details
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # # Load dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = get_dataset(script_args.dataset_name, script_args.dataset_config)

    ### Load Tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    


    report_answer_length_ranges(
        dataset     = dataset,
        answer_key  = "solution",     # or "answer" if that is the column name
        tokenizer   = tokenizer,
        buckets     = (2000, 6000, 10000, 12000)  # customise as you like
    )

    # Define reward functions
    REWARD_FUNCS_REGISTRY = {
        "confidence_based_reward": confidence_based_reward,
        "entropy": entropy_based_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format dataset into conversation
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    ### Initialize Model Keyword Arguments
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    ### Define PEFT Configuration with LoRA
    peft_config = LoraConfig(
        r=128,  # Rank of the low-rank matrices
        lora_alpha=128,  # Scaling factor for LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target modules for LoRA
        lora_dropout=0.05,  # Dropout rate for LoRA layers
        bias="none",  # Bias configuration
        task_type="CAUSAL_LM",  # Task type for causal language modeling
    )

    ### Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,  # Pass the PEFT configuration
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ### Training Loop
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ### Save Model and Create Model Card
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ### Evaluate
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ### Push to Hub
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)