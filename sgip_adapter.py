import scipy
import PIL
import numpy as np
import torch
from attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from attention_processor import AttnProcessor, IPAttnProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import CLIPTextModel
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from transformers.models.clip.modeling_clip import _make_causal_mask, _expand_mask
from diffusers import ControlNetModel, UNet2DConditionModel
import torch.nn as nn
from utils import is_torch2_available


class ControlNetModelCat(ControlNetModel):
    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate a [`ControlNetModel`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`ControlNetModel`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels*2,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        #if load_weights_from_unet:
        #    controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
        #    controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        #    controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
        #    if controlnet.class_embedding:
        #        controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())
        #    controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        #    controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
        
        if load_weights_from_unet:
            unet_sd = unet.state_dict()
            scratch_sd = controlnet.state_dict()
            init_sd = {}
            init_with_new_zero = set()
            init_with_scratch = set()
            for key in scratch_sd:
                if key in unet_sd:
                    this, target = scratch_sd[key], unet_sd[key]
                    if this.size() == target.size():
                        init_sd[key] = target.clone()
                    else:
                        d_ic = this.size(1) - target.size(1)
                        oc, _, h, w = this.size()
                        zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                        init_sd[key] = torch.cat((target, zeros), dim=1)
                        init_with_new_zero.add(key)
                else:
                    init_sd[key] = scratch_sd[key].clone()
                    init_with_scratch.add(key)
            controlnet.load_state_dict(init_sd, strict=True)

        return controlnet
   
class CLIPTextModelWrapper(CLIPTextModel):
    # Adapted from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/clip/modeling_clip.py#L812
    # Modified to accept precomputed token embeddings "input_token_embs" as input or calculate them from input_ids and return them.
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_token_embs: Optional[torch.Tensor] = None,
        return_token_embs: Optional[bool] = False,
    ) -> Union[Tuple, torch.Tensor, BaseModelOutputWithPooling]:

        if return_token_embs:
            return self.text_model.embeddings.token_embedding(input_ids)
    
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        output_attentions = output_attentions if output_attentions is not None else self.text_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.text_model.config.use_return_dict
    
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
    
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    
        hidden_states = self.text_model.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=input_token_embs)
    
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
    
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)
    
        if self.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
    
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class TextProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        #x = x.reshape(-1, self.cross_attention_dim)
        x = self.norm(x)
        return x
    

def attn_processors_deal(unet):
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    return attn_procs

def attn_processors_deal_lora(unet):
    attn_procs = {}
    lora_rank = 128
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            attn_procs[name].load_state_dict(weights, strict=False)
    return attn_procs


    
class SGIPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, controlnet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=4,)
        
        self.text_proj_model = TextProjModel(
            cross_attention_dim=768,
            id_embeddings_dim=768)
        
    def forward(self, noisy_latents, latents_lq, timesteps, encoder_hidden_states, image_embeds, emocas):
        # image_embeds
        # print(encoder_hidden_states.shape, image_embeds.shape)
        control_net_input = torch.cat([noisy_latents, latents_lq], dim=1)
        ip_tokens = self.image_proj_model(image_embeds)
        #print(encoder_hidden_states.shape, ip_tokens.shape)
        encoder_hidden_states = self.text_proj_model(encoder_hidden_states)
        # encoder_hidden_states为clip输出的文本
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        #control_net_input = noisy_latents + latents_lq
        # ControlNet encoder_hidden_states层的输入为ip_tokens
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_net_input,
            timesteps,
            encoder_hidden_states=ip_tokens,
            controlnet_cond = emocas,  # 传入条件图
            return_dict=False,
        )
        
        # Predict the noise residual  encoder_hidden_states输入到unet中表示解耦交叉注意力
        noise_pred = self.unet(noisy_latents, timesteps, 
                               encoder_hidden_states = encoder_hidden_states,                
                               down_block_additional_residuals = down_block_res_samples,
                                mid_block_additional_residual = mid_block_res_sample).sample
        
        return noise_pred

    
        