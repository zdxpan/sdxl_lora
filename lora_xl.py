import numpy as np
import re
import torch
import torch.nn.functional as F
import itertools

from tqdm import tqdm, trange
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from transformers import CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from PIL import Image

def create_add_time_ids(
    original_size,
    target_top_left,
    target_size,
    requires_aesthetics_score=False,
    aesthetic_score=6.0,
    negative_aesthetic_score=2.5
):
    if requires_aesthetics_score:
        add_time_ids = list(original_size + target_top_left + (aesthetic_score,))
        add_neg_time_ids = list(original_size + target_top_left + (negative_aesthetic_score,))
    else:
        add_time_ids = list(original_size + target_top_left + target_size)
        add_neg_time_ids = list(original_size + target_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_neg_time_ids = torch.tensor([add_neg_time_ids])
    return add_time_ids, add_neg_time_ids


def make_text_embeddings(tokenizer, tokenizer_2, text_encoder, text_encoder_2, prompt, device):
    tokenizers = [tokenizer, tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2]

    prompt_embeds_list = []
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_input = tokenizer(
            [prompt],
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_embeds = text_encoder(
            text_input.input_ids.to(device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    add_text_embeds = pooled_prompt_embeds

    return prompt_embeds, add_text_embeds


def create_np_image(image, normalize=True):
    image = image.convert('RGB')
    # print(
    #     f'>> DEBUG: writing the image to img.png'
    # )
    # image.save('img.png')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    if normalize:
        image = 2.0 * image - 1.0
    return image


def unet_attn_processors_state_dict(unet):
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict

# -- 训练SDXL lora  应用于  ,脚本开发 by nanqiao~

def main():
    model_root = '/home/dell/workspace/models/stable-diffusion-xl-base-1.0'
    output_dir = 'jlout'
    device = 'cuda'
    dtype = torch.float32
    steps = 150
    prompt = 'jialin'
    img_s = [#"./jl_test/jl_test_face.png", 
            "./jl_test/jl_test_pure.png"]
    # img_ = "./jl_test/jl_test_pure.png"
    # sz = (512,512) 
    sz = (768, 768) 
    images = [Image.open(img_) for img_ in img_s]
    images = [image.resize(sz, resample=Image.Resampling.LANCZOS) for image in images]
    # image = image.resize(sz, resample=Image.Resampling.LANCZOS)

    scheduler = DDPMScheduler.from_pretrained(
        model_root,
        subfolder='scheduler',
        torch_dtype=dtype,
        use_safetensors=True,
        variant='fp16'
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_root,
        subfolder='unet',
        torch_dtype = dtype,
        use_safetensors=True,
        variant='fp16'
    ).to(device)

    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_root,
        subfolder='text_encoder',
        torch_dtype=torch.float32,
        use_safetensors=True,
        variant='fp16'
    ).to(device)

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_root,
        subfolder='text_encoder_2',
        torch_dtype=torch.float32,
        use_safetensors=True,
        variant='fp16'
    ).to(device)

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        model_root,
        subfolder='tokenizer',
        torch_dtype=dtype,
        use_safetensors=True,
        variant='fp16'
    )

    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_root,
        subfolder='tokenizer_2',
        torch_dtype=dtype,
        use_safetensors=True,
        variant='fp16'
    )

    vae = AutoencoderKL.from_pretrained(
        model_root,
        subfolder='vae_fp16_fix',
        torch_dtype = dtype,
        use_safetensors=True,
        variant='fp16'
    ).to(device)
    
    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_processor_class = (
            LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        )
        module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        unet_lora_attn_procs[name] = module.to(device)
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)
    
    text_lora_parameters_1 = LoraLoaderMixin._modify_text_encoder(text_encoder_1, dtype=torch.float32)
    text_lora_parameters_2 = LoraLoaderMixin._modify_text_encoder(text_encoder_2, dtype=torch.float32)
    
    optimizer = torch.optim.AdamW(
        itertools.chain(unet_lora_parameters, text_lora_parameters_1, text_lora_parameters_2),
        lr=0.0005
    )

    add_time_ids, _ = create_add_time_ids(
        images[0].size,
        (0, 0),
        images[0].size
    )
    add_time_ids = add_time_ids.to(device, dtype=dtype)


    pbar = tqdm(range(steps))
    pbar.set_description('Steps')
    inx = 0 
    for i in pbar: 
        # Sample noise that we'll add to the latents
        # swaper image
        image = images[inx % len(images)]
        inx += 1
        init_image = create_np_image(image).to(device, dtype=dtype)
        init_latent = vae.encode(init_image).latent_dist.sample()
        init_latent = init_latent * vae.config.scaling_factor

        noise = torch.randn_like(init_latent, device=device)
        bsz = init_latent.shape[0]
        
        # Sample a random timestep for each image
        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device)
        t = t.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latent = scheduler.add_noise(init_latent, noise, t)

        prompt_embeds, add_text_embeds = make_text_embeddings(
            tokenizer_1,
            tokenizer_2,
            text_encoder_1,
            text_encoder_2,
            prompt,
            device
        )

        # Predict the noise residual
        noise_pred = unet(
            noisy_latent,
            t,
            encoder_hidden_states=prompt_embeds.to(dtype),
            added_cond_kwargs={'text_embeds': add_text_embeds.to(dtype), 'time_ids': add_time_ids},
            return_dict=False
        )[0]

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')

        optimizer.zero_grad() # Zero the gradients
        loss.backward()       # Compute gradients   
        optimizer.step()      # Update weights
        
        pbar.set_postfix(loss=loss.item())
    

    unet_lora_layers = unet_attn_processors_state_dict(unet)
    text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder_1.to(torch.float32))
    text_encoder_2_lora_layers = text_encoder_lora_state_dict(text_encoder_2.to(torch.float32))
    
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=text_encoder_lora_layers,
        text_encoder_2_lora_layers=text_encoder_2_lora_layers,
    )


if __name__ == "__main__":
    main()
