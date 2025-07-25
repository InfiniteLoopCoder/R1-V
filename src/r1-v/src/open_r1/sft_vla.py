# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models (VIDEO ONLY).

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name path/to/your/dataset.json \
    --dataset_root_path /path/to/your/videos \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill-VideoOnly


    # --- 在这里添加PEFT/LoRA参数 ---
    --use_peft True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj"
"""

import logging
import os
import sys
from functools import partial

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)

DATASET_ROOT_PATH = "/home/yaofeng/R1-V/Robot-VLA-R1"

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})



processor = None


def convert_example(example, dataset_root_path=None):
    """
    correct example into "messages" for video-only data
    """
    messages = []
    if "system" in example:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": example["system"]}],
        })
    else:
        SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
        )
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        })

    thinking = example.get("process")
    problem = example.get("problem")
    solution = example.get("solution")
    video_relative_path = example.get("path")
    
    user_content = [{"type": "text", "text": problem}]
    # Video-only logic: only check for and process video paths.
    if video_relative_path and dataset_root_path:
        full_video_path = os.path.join(dataset_root_path, video_relative_path)
        if os.path.exists(full_video_path):
            user_content.append({"type": "video", "video": full_video_path})
        else:
            logger.warning(f"Video path does not exist, skipping: {full_video_path}")

    messages.append({
        "role": "user",
        "content": user_content
    })
    
    messages.append({
        "role": "assistant",
        "content": f"{thinking}\n\n{solution}",
    })
    
    example["messages"] = messages
    return example
    
def collate_fn(examples, dataset_root_path=None):
    # 1. Convert all examples to the "messages" format.
    converted_examples = [convert_example(ex, dataset_root_path=dataset_root_path) for ex in examples]

    # 2. Apply chat template to get the text for each example.
    texts = [
        processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        for ex in converted_examples
    ]
    
    # 3. Process all video inputs from the batch, ignoring images.
    video_inputs = []
    for ex in converted_examples:
        try:
            # process_vision_info returns (images, videos). We ignore images with _.
            _, vids = process_vision_info(ex["messages"])
        except Exception:
            vids = [] # If video processing fails, treat it as no video.
            
        if vids:
            video_inputs.extend(vids)

    # 4. Use the processor to tokenize text and prepare video tensors.
    batch = processor(
        text=texts,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    # 5. Create labels for language model training.
    labels = batch["input_ids"].clone()
    # Mask out padding tokens and any residual image/video tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# def collate_fn(examples, dataset_root_path=None):
#     # 1. Convert all examples to the "messages" format.
#     converted_examples = [convert_example(ex, dataset_root_path=dataset_root_path) for ex in examples]

#     # 2. Apply chat template to get the text for each example.
#     texts = [
#         processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
#         for ex in converted_examples
#     ]
    
#     # 3. Process all video inputs from the batch, ignoring images.
#     video_inputs = []
#     for ex in converted_examples:
#         # process_vision_info returns (images, videos). We ignore images with _.
#         _, vids = process_vision_info(ex["messages"])
#         if vids:
#             video_inputs.extend(vids)

#     # 4. Use the processor to tokenize text and prepare video tensors.
#     batch = processor(
#         text=texts,
#         videos=video_inputs if video_inputs else None,
#         return_tensors="pt",
#         padding=True,
#     )

#     # 5. Create labels for language model training.
#     labels = batch["input_ids"].clone()
#     # Mask out padding tokens and any residual image/video tokens
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
#     labels[labels == image_token_id] = -100
#     batch["labels"] = labels

#     return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    data_files = {}
    if script_args.dataset_train_split and os.path.exists(script_args.dataset_train_split):
        data_files['train'] = script_args.dataset_train_split
    elif script_args.dataset_name and os.path.exists(script_args.dataset_name):
        data_files['train'] = script_args.dataset_name

    if script_args.dataset_test_split and os.path.exists(script_args.dataset_test_split):
        data_files['test'] = script_args.dataset_test_split

    if data_files:
        logger.info(f"Loading dataset from JSON files: {data_files}")
        dataset = load_dataset("json", data_files=data_files)
    else:
        logger.info(f"Loading dataset from Hugging Face Hub: {script_args.dataset_name}")
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    # from transformers import Qwen2VLForConditionalGeneration
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path, **model_kwargs
    # )
    ############################
    # Initialize the SFT Trainer
    ############################
    # Create a partial function for the data collator to pass the dataset_root_path
    data_collator_with_path = partial(collate_fn, dataset_root_path=DATASET_ROOT_PATH)

    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        tokenizer=processor.tokenizer,
        data_collator=data_collator_with_path,
        peft_config=get_peft_config(model_args)
    )

    ###############
    # Training loop
    ###############
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

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V", "video-only"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
