nohup accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
train.py \
  --pretrained_model_name_or_path="/root/autodl-tmp/project/SGIP-dev-diffuser/pretrain" \
  --data_json_file="/root/autodl-tmp/project/SGIP-dev-v2/data_json/PIE_clean_s1expos_train.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs/concat_text_pro_g2_resume" \
  --save_steps=40000 \
  --num_train_epochs=6 \
  --ckpt_path="/root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs/concat_text_pro_g2/checkpoint-120000/pytorch_model.bin" &