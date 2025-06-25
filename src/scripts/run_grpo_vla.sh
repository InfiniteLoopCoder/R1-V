export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_3b.txt"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir data/Qwen2.5-VL-3B-Instruct-GRPO \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /home/yaofeng/R1-V/Robot-VLA-R1/merged_problems.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2.5-VL-3B-Instruct-GRPO \
    --save_steps 100 \
    --save_only_model true