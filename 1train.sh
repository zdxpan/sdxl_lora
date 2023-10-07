# -- 使用扣除背景的图训练~
# conda activate sd
nohup accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=/home/dell/workspace/models/stable-diffusion-xl-base-1.0  \
  --instance_data_dir=/home/dell/workspace/xl_lora/zdxpure \
  --output_dir=zdxpure \
  --instance_prompt="zdx" \
  --resolution=768 \
  --crop_size=768 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --checkpointing_steps=100 \
  --num_train_epochs=800 \
  --max_train_steps=800 \
  --validation_epochs=50 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --validation_prompt="zdx, in singing" \
  --seed="0"  > 1lora.log 2>&1 &


# -- 使用原图训练
# accelerate launch train_dreambooth_lora_sdxl.py \
#   --pretrained_model_name_or_path=/home/dell/workspace/models/stable-diffusion-xl-base-1.0  \
#   --instance_data_dir=/home/dell/workspace/xl_lora/zdx \
#   --output_dir=zdx \
#   --instance_prompt="zdxpan" \
#   --resolution=1024 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=2 \
#   --checkpointing_steps=100 \
#   --num_train_epochs=100 \
#   --learning_rate=1e-4 \
#   --report_to="wandb" \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=200 \
#   --validation_prompt="zdxpan2 are singing" \
#   --validation_epochs=60 \
#   --seed="0"
