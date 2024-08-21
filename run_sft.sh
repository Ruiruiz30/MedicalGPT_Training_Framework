CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 2 supervised_finetuning.py \
    --model_type baichuan \
    --model_name_or_path .\pretrainModel \
    --train_file_dir ./data/finetune/dataformedical  \
    --validation_file_dir ./data/finetune/dataformedical  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --template_name qwen \
     --qlora True \
    --load_in_8bit True \
    --torch_dtype bfloat16 \
    --optim paged_adamw_32bit \
    --max_train_samples -1 \
    --max_eval_samples 500000 \
    --model_max_length 6144 \
    --num_train_epochs 50 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 1 \
    --preprocessing_num_workers 20 \
    --output_dir outputs-sft-qwen-v1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --flash_attn \
    -shift_attn \


