# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python kto.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_strategy steps\
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python kto.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 5e-7 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_strategy steps\
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import wandb

from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    ScriptArguments,
    get_peft_config,
    setup_chat_format,
)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    wandb.init(project='DAA_Analysis', name=f'{model_args.model_name_or_path}', group=f'KTO',)

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)

    # Initialize the KTO trainer
    trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].select(range(5000)),
        eval_dataset=dataset[script_args.dataset_test_split].select(range(500)) if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    wandb.finish()