# Exploring Alternatives to RLHF
In this repo we analyze different DAA algorithms for LLM alignment to human preferences.
Aligning large language models (LLMs) with human preferences is essential for their safe and effective use in real-world applications. Traditional Reinforcement Learning from Human Feedback (RLHF) methods, while effective, face challenges such as instability, high computational costs, and reliance on explicit reward models. This project investigates four RLHF alternatives—Direct Policy Optimization (DPO), Kahneman-Tversky Optimization (KTO), Odds Ratio Preference Optimization (ORPO), and Simple Preference Optimization (SimPO).

## Experiment Design

Algorithms Evaluated:
1. DPO
2. KTO
3. ORPO
4. SimPO

We evaluate on two models to understand how the size of the policy (in Billion parameters) effects the ability to learn the preferences:
1. Qwen/Qwen2-0.5B-Instruct
2. EleutherAI/pythia-2.8b

We again evaluate on two popular preference datasets for better generalizability of our results:
1. Anthropic HH-RLHF
2. TL;DR Preference Dataset

We use robust metrics described in more detail in the report:
1. Loss: negative log-likelihood of the correct output
2. Reward Margin:  difference between the rewards assigned to "chosen" and "rejected" responses
3. LLM as Judge Win Rate: we use the Gemini model to compute the win rate between the models response and the ground truth

## Steps to run

### Arguments

DATASET_NAME: Specify the dataset to use. For our study, we use the following 2 options:
1. trl-internal-testing/hh-rlhf-trl-style
2. trl-internal-testing/tldr-preference-trl-style

MODEL_NAME: Specify the model to use. For our study, we use the following 2 options:
1. Qwen/Qwen2-0.5B-Instruct
2. EleutherAI/pythia-2.8b

OUTPUT_DIR: Directory to store the learned LORA adapters after alignment

This same OUTPUT_DIR can be passed as the LORA_PATH for evaluation:
1. LORA_PATH = OUTPUT_DIR

### DPO

Command to 'train' (align) the policy using the DPO algorithm:
```
python dpo.py \
    --dataset_name DATASET_NAME \
    --model_name_or_path MODEL_NAME \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 20 \
    --output_dir outputs \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --report_to wandb
    --push_to_hub
```

### KTO

### ORPO

### SimPO

### Win Rate Calculation
We use the Gemini-1.5-Flash model as the Judge to compute the win rates.

1. Add your GOOGLE_API_KEY to the environment as we use Gemini Free API to score the responses using the LLM as a judge paradigm
```
export GOOGLE_API_KEY=''
```
2.
Run Command to compute the Gemini Win Rates for the policy aligned using your algorithm:
```
python3 eval.py \
  --dataset_name DATASET_NAME \
  --model_name MODEL_NAME \
  --lora_path LORA_PATH
```
