import os
import argparse
from pathlib import Path
import itertools
import time, datetime
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, DDIMScheduler
from transformers import CLIPTokenizer
from sgip_adapter import attn_processors_deal, attn_processors_deal_lora
from sgip_adapter import SGIPAdapter, CLIPTextModelWrapper, ControlNetModelCat
from sgip_dataset import TrainDataset, collate_fn
import numpy as np
from PIL import Image
from accelerate.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default="/root/autodl-tmp/project/SGIP-dev-v2/data_json/PIE_clean_s1expos_val_lite_490.json",
        help="val data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=20000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--ckpt_path", type=str, default= "")
   
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    set_seed(666, device_specific=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModelWrapper.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModelCat.from_unet(unet)
    
    ddim_noise_scheduler = DDIMScheduler(
                                num_train_timesteps=1000,
                                beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule="scaled_linear",
                                clip_sample=False,
                                set_alpha_to_one=False,
                                steps_offset=1,)
    
    num_inference_steps = 50
    
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # controlnet.requires_grad_(True)
    #image_encoder.requires_grad_(False)
    #unet.to(accelerator.device, dtype=weight_dtype)
    #attn_procs = attn_processors_deal(unet)
    
    attn_procs = attn_processors_deal_lora(unet)
    unet.set_attn_processor(attn_procs)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    #controlnet.to(accelerator.device, dtype=weight_dtype)
    #image_proj_model.to(accelerator.device, dtype=weight_dtype)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = SGIPAdapter(unet, controlnet)
    
    if args.ckpt_path:
        ip_adapter.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
        ip_adapter.train().to(accelerator.device)
    
    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), 
                                    ip_adapter.text_proj_model.parameters(), 
                                    adapter_modules.parameters(), 
                                    ip_adapter.controlnet.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    opt_path = "/root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs/concat_text_pro_g2/checkpoint-120000/optimizer.bin"
    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    
    # dataloader
    train_dataset = TrainDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # val dataloader
    val_dataset = TrainDataset(args.val_json, tokenizer=tokenizer, size=args.resolution)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    # 5. Prepare timesteps
    ddim_noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    inf_timesteps = ddim_noise_scheduler.timesteps
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader, val_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader, val_dataloader)
    global_step = 0
    log_file = os.path.join(args.output_dir, "log.txt")
        
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents_gt = vae.encode(batch["gt"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents_gt = latents_gt * vae.config.scaling_factor
                    latents_lq = vae.encode(batch["lq"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents_lq = latents_lq * vae.config.scaling_factor
                    
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents_gt)
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents_gt.shape[0],), device=latents_gt.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                latents_z0 = noise_scheduler.add_noise(latents_gt, noise, timesteps)
                image_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                
                with torch.no_grad():
                    input_ids = batch["input_id"]
                    arcface_token_id = tokenizer.encode("id", add_special_tokens=False)[0]
                    face_embs_padded = F.pad(image_embeds, (0, text_encoder.config.hidden_size-512), "constant", 0)
                    token_embs = text_encoder(input_ids=input_ids, return_token_embs=True)
                    token_embs[input_ids==arcface_token_id] = face_embs_padded
                    encoder_hidden_states = text_encoder(input_ids=input_ids.to(accelerator.device), input_token_embs=token_embs)[0]
                    
                emoca = batch["emoca"].to(accelerator.device, dtype=weight_dtype)
                noise_pred = ip_adapter(latents_z0, latents_lq, timesteps, encoder_hidden_states, image_embeds, emoca)
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and step % 200 == 0:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
                    
                    p_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # 打开文件进行写入
                    with open(log_file, 'a') as f:  # 'a'模式表示附加写入，不会覆盖文件内容
                        f.write("Time:{}, Steps:{}, Epoch:{}, avg_loss:{} \n".format(p_time, global_step, epoch, avg_loss))
                    
            global_step += 1
            total_step = len(train_dataloader) * args.num_train_epochs
            if global_step % 20000 == 0 or global_step == total_step - 1:
                ip_adapter.eval()
                save_path = os.path.join(args.output_dir, "inf", str(global_step))
                os.makedirs(save_path, exist_ok=True)
                with torch.no_grad():
                    for j, val_batch in enumerate(val_dataloader):
                        #if j > 30 and global_step < total_step - 1:
                        #    break
                        # get id_emb
                        image_embeds_val = val_batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                        # get text_emb
                        input_ids_val = val_batch["input_id"].to(accelerator.device, dtype=torch.long)
                        arcface_token_id_val = tokenizer.encode("id", add_special_tokens=False)[0]
                        face_embs_padded_val = F.pad(image_embeds_val, (0, text_encoder.config.hidden_size-512), "constant", 0)
                        token_embs_val = text_encoder(input_ids=input_ids_val, return_token_embs=True)
                        token_embs_val[input_ids_val==arcface_token_id_val] = face_embs_padded_val
                        encoder_hidden_states_val = text_encoder(input_ids=input_ids_val, input_token_embs=token_embs_val)[0]
                        # get emoca
                        emoca_val = val_batch["emoca"].to(accelerator.device, dtype=weight_dtype)
                        # get lq latents
                        latents_lq_val = vae.encode(val_batch["lq"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        latents_lq_val = latents_lq_val * vae.config.scaling_factor
                        # get zt latent
                        latents_val = torch.randn_like(latents_lq_val)
                        # denoising
                        for t_val in inf_timesteps:
                            noise_pred = ip_adapter(latents_val, latents_lq_val, t_val, encoder_hidden_states_val, image_embeds_val, emoca_val)
                            latents_val = ddim_noise_scheduler.step(noise_pred, t_val, latents_val).prev_sample
                        # vae decoder
                        pred_images = vae.decode(latents_val.to(accelerator.device, dtype=weight_dtype) / vae.config.scaling_factor)[0][0]
                        
                        pred_images_2 = (pred_images / 2 + 0.5).clamp(0, 1)
                        pred_images_2 = pred_images_2.cpu().permute(1, 2, 0).numpy() * 255
                        pred_images_2 = Image.fromarray(pred_images_2.astype(np.uint8))
                        pred_images_2.save(os.path.join(save_path, val_batch["img_name"][0]))                      
                        
                ip_adapter.train()
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
