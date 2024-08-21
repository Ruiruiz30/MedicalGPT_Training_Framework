@echo off
set CUDA_VISIBLE_DEVICES=0

call python  supervised_finetuning.py ^
    --model_type baichuan ^
    --model_name_or_path .\pretrainModel ^
    --train_file_dir .\data\finetune\dataformedical ^
    --validation_file_dir .\data\finetune\dataformedical ^
    --do_train ^
    --do_eval ^
    --template_name qwen ^
    --use_peft True ^
    --model_max_length 6144 ^
    --num_train_epochs 50 ^
    --learning_rate 2e-5 ^
    --warmup_ratio 0.05 ^
    --weight_decay 0.05 ^
    --logging_strategy steps ^
    --logging_steps 10 ^
    --eval_steps 10 ^
    --evaluation_strategy steps ^
    --save_steps 500 ^
    --save_strategy steps ^
    --save_total_limit 13 ^
    --gradient_accumulation_steps 1 ^
    --preprocessing_num_workers 4 ^
    --output_dir outputs-sft-qwen-v1 ^
    --overwrite_output_dir ^
    --ddp_timeout 30000 ^
    --logging_first_step True ^
    --target_modules all ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --lora_dropout 0.05 ^
    --torch_dtype float16 ^
    --fp16 ^
    --device_map auto ^
    --report_to tensorboard ^
    --ddp_find_unused_parameters False ^
    --gradient_checkpointing True ^
    --cache_dir .\cache ^
    --flash_attn ^
    --shift_attn ^
    --load_in_8bit True


pause
