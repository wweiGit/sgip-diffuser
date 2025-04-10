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
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer
from sgip_adapter import attn_processors_deal, attn_processors_deal_lora
from sgip_adapter import SGIPAdapter, CLIPTextModelWrapper, ControlNetModelCat
from sgip_dataset import TrainDataset, collate_fn
import numpy as np
from PIL import Image
from accelerate.utils import set_seed
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/root/autodl-tmp/project/SGIP-dev-diffuser/pretrain",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--val_json", type=str)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    set_seed(666, device_specific=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModelWrapper.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModelCat.from_unet(unet)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    ddim_noise_scheduler = DDIMScheduler(
                                num_train_timesteps=1000,
                                beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule="scaled_linear",
                                clip_sample=False,
                                set_alpha_to_one=False,
                                steps_offset=1,
                            )
    num_inference_steps = args.steps 
    
    # init adapter modules
    attn_procs = attn_processors_deal_lora(unet)
    unet.set_attn_processor(attn_procs)
    ip_adapter = SGIPAdapter(unet, controlnet)
    ip_adapter.load_state_dict(torch.load(args.ckpt_path))
    ip_adapter.eval().to(accelerator.device, dtype=weight_dtype)
    #image_proj_model.to(accelerator.device, dtype=weight_dtype)
    #controlnet.to(accelerator.device, dtype=weight_dtype)
    # val dataloader
    val_dataset = TrainDataset(args.val_json, tokenizer=tokenizer, size=args.resolution)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=args.dataloader_num_workers,
    )
    # 5. Prepare timesteps
    ddim_noise_scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    inf_timesteps = ddim_noise_scheduler.timesteps
    os.makedirs(args.save_path, exist_ok=True)

    with tqdm(total=len(val_dataloader), desc="Inference", position=0) as pbar:
        with torch.no_grad():
            for i, val_batch in tqdm(enumerate(val_dataloader)):
                # get id_emb
                image_embeds_val = val_batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                # get text_emb
                input_ids_val = val_batch["input_id"].to(accelerator.device, dtype=torch.long)
                arcface_token_id_val = tokenizer.encode("id", add_special_tokens=False)[0]
                face_embs_padded_val = F.pad(image_embeds_val, (0, text_encoder.config.hidden_size-512), "constant", 0)
                token_embs_val = text_encoder(input_ids=input_ids_val, return_token_embs=True)
                token_embs_val[input_ids_val==arcface_token_id_val] = face_embs_padded_val
                encoder_hidden_states_val = text_encoder(input_ids=input_ids_val.to(accelerator.device), input_token_embs=token_embs_val)[0]
                # get emoca
                emoca_val = val_batch["emoca"].to(accelerator.device, dtype=weight_dtype)
                # get lq latents
                latents_lq_val = vae.encode(val_batch["lq"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents_lq_val = latents_lq_val * vae.config.scaling_factor
                # get zt latent
                latents_val = torch.randn_like(latents_lq_val).to(accelerator.device, dtype=weight_dtype)
                # denoising
                for t_val in inf_timesteps:
                    encoder_hidden_states_val.to(accelerator.device, dtype=weight_dtype)
                    noise_pred = ip_adapter(latents_val, latents_lq_val, t_val, encoder_hidden_states_val, image_embeds_val, emoca_val)
                    latents_val = ddim_noise_scheduler.step(noise_pred, t_val, latents_val).prev_sample
                # vae decoder
                pred_images = vae.decode(latents_val.to(accelerator.device, dtype=weight_dtype) / vae.config.scaling_factor)[0]
                pred_images = (pred_images / 2 + 0.5).clamp(0, 1)
                pred_images = pred_images.cpu().permute(0, 2, 3, 1).numpy() * 255
            
                for j, sample in enumerate(pred_images):
                    img_save_path = os.path.join(args.save_path, val_batch["img_name"][j])
                    sample = Image.fromarray(sample.astype(np.uint8))
                    sample.save(img_save_path)    
                                
                pbar.update()
                        
                    
if __name__ == "__main__":
    main()    
