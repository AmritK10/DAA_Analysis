
"""
# Full training
python orpo.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# USE THIS
# LoRA:
python orpo.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 40 \
    --output_dir Pythia3B \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --report_to wandb
"""

"""
python orpo.py \
    --dataset_name trl-internal-testing/tldr-preference-trl-style \
    --dataset_test_split validation \
    --model_name_or_path EleutherAI/pythia-2.8b \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 20 \
    --output_dir Pythia-tldr \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --report_to wandb
"""


import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    DPOConfig,
    DPOTrainer,
    ORPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import os
os.environ['HF_HOME'] = '/home/hice1/tbhotia3/scratch/cache/'

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    wandb.init(project='DAA_Analysis-PYTHIA-TLDR', name=f'{model_config.model_name_or_path}', group=f'PYTHIA-TLDR',)

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        device_map='auto',
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    print(peft_config)
    if peft_config is None:
        print("NONEEEE")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)

    ##########
    # Training
    ################
    trainer = ORPOTrainer(
        model,
        # ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].select(range(10000)),
        eval_dataset=dataset[script_args.dataset_test_split].select(range(1000)) if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

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
