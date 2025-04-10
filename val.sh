accelerate launch --num_processes 1 \
val.py \
    --save_path /root/autodl-tmp/project/SGIP-dev-diffuser/inf/concat_b2_resume \
    --val_json /root/autodl-tmp/project/SGIP-dev-v2/data_json/PIE_clean_s1expos_val.json \
    --ckpt_path /root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs/concat_text_pro_g2_resume/checkpoint-80000/pytorch_model.bin 