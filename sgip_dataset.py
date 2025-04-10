import torch
from torchvision import transforms
import json
from PIL import Image
import torch.nn.functional as F
import os 
import random 
@torch.no_grad()

def prompt_face_embs(text, tokenizer):
    
    input_ids = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids


    return input_ids

def collate_fn(data):
    lq_images = torch.stack([example["lq"] for example in data])
    gt_images = torch.stack([example["gt"] for example in data])
    emocas = torch.stack([example["emoca"] for example in data])
    input_ids = torch.cat([example["input_id"] for example in data], dim=0)
    face_id_embeds = torch.stack([example["face_id_embed"] for example in data])
    img_name = [example["img_name"] for example in data]
    return {
        "lq": lq_images,
        "gt": gt_images,
        "emoca": emocas,
        "input_id": input_ids,
        "face_id_embed": face_id_embeds,
        "img_name": img_name
    }
    
# TrainDataset
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size

        #[{"img_name": "001_01_01_090_06_cropped.png", 
        #"img_path": "/root/autodl-tmp/dataset/DATA/MTPIE/train_val_split/train/lq_64_f128/001_01_01_090_06_cropped.png", 
        #"clean_path": "/root/autodl-tmp/project/SGIP-dev/inf_results/stage1_expos/001_01_01_090_06_cropped.png",
        #"gt_path": "/root/autodl-tmp/dataset/DATA/MTPIE/train_val_split/train/gt/001_01_01_090_06_cropped.png", 
        #"id_path": "/root/autodl-tmp/dataset/DATA/MTPIE/train_val_split/train/id_emb_ada_s1expos/001_01_01_090_06_cropped.bin", 
        #"emoca_path": "/root/autodl-tmp/dataset/DATA/MTPIE/train_val_split/train/3dmm/001_01_01_051_08_cropped.png"}]
        
        self.data = json.load(open(json_file)) 

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __getitem__(self, idx):
        item = self.data[idx] 
        # read image
        lq_image = Image.open(item["clean_path"])
        gt_image = Image.open(item["gt_path"])
        emoca_image = Image.open(item["emoca_path"])
        
        lq_image = self.transform(lq_image.convert("RGB"))
        gt_image = self.transform(gt_image.convert("RGB"))
        emoca_image = self.transform(emoca_image.convert("RGB"))
        
        face_id_embed = torch.load(item["id_path"], map_location="cpu").detach()
        #print("face_id_embed requires_grad:", face_id_embed.requires_grad)
        face_id_embed = face_id_embed / torch.norm(face_id_embed)   # normalize embedding

        # get text and prompt_embeds
        text = "photo of a id person"
        input_ids = prompt_face_embs(text, self.tokenizer)
        
        return {
            "lq": lq_image,
            "gt": gt_image,
            "emoca": emoca_image,
            "face_id_embed": face_id_embed,
            "input_id": input_ids,
            "img_name": item["img_name"]
        }

    def __len__(self):
        return len(self.data)