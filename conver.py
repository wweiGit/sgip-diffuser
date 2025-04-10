import torch
ckpt = "/root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs_new/checkpoint-80000/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
unet_sd = {}
control_sd = {}
for k in sd:
    if k.startswith("unet"):
        unet_sd[k.replace("unet.", "")] = sd[k]
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("controlnet"):
        control_sd[k.replace("controlnet.", "")] = sd[k]

torch.save({"unet": unet_sd, "image_proj": image_proj_sd, "controlnet": control_sd}, "/root/autodl-tmp/project/SGIP-dev-diffuser/work_dirs_new/checkpoint-80000/sgip_80000.bin")