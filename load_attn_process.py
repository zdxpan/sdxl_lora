
import torch
from collections import defaultdict
from stable_diffusion import load_state_dict
import torch.nn.functional as F
# from safetensors.torch import save_file
from diffusers import (AutoencoderKL,    DDPMScheduler,    UNet2DConditionModel,    UniPCMultistepScheduler)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    SlicedAttnAddedKVProcessor,
    XFormersAttnProcessor,
)

lora = load_state_dict("/home/dell/workspace/models/pixel-art-xl.safetensors")
unet = UNet2DConditionModel.from_config("/home/dell/workspace/models/stable-diffusion-xl-base-1.0/unet/config.json")

state_dict = lora
# fill attn processors
attn_processors = {}
network_alpha = 1.0      #  kwargs["network_alpha"]

is_lora = all("lora" in k for k in state_dict.keys())
is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())
# is_lora
if is_lora:
    is_new_lora_format = all(
        key.startswith(unet.unet_name) or key.startswith(unet.text_encoder_name) for key in state_dict.keys()
    )
    if is_new_lora_format:
        # Strip the `"unet"` prefix.
        is_text_encoder_present = any(key.startswith(unet.text_encoder_name) for key in state_dict.keys())
        if is_text_encoder_present:
            warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
            # warnings.warn(warn_message)
            print(warn_message)
        unet_keys = [k for k in state_dict.keys() if k.startswith(unet.unet_name)]
        state_dict = {k.replace(f"{unet.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

    lora_grouped_dict = defaultdict(dict)
    for key, value in state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processor = unet
        for sub_key in key.split("."):
            attn_processor = getattr(attn_processor, sub_key)

        if isinstance(
            attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)
        ):
            cross_attention_dim = value_dict["add_k_proj_lora.down.weight"].shape[1]
            attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
            if isinstance(attn_processor, (XFormersAttnProcessor, LoRAXFormersAttnProcessor)):
                attn_processor_class = LoRAXFormersAttnProcessor
            else:
                attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )

        attn_processors[key] = attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            network_alpha=network_alpha,
        )
        attn_processors[key].load_state_dict(value_dict)

# # set correct dtype & device
# attn_processors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in attn_processors.items()}

# # set layers
# unet.set_attn_processor(attn_processors)
