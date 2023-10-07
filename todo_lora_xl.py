"""
# 因为训练完成后，跑图无效。因此前缀改名为TODO
accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337
"""

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

from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from accelerate.logging import get_logger

logger = get_logger(__name__)

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_num=None,
        size=1024,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        print(self.instance_images_path[0], type(self.instance_images_path[0]))
        # print(dir(self.instance_images_path[0]))
        print(self.instance_images_path[0].suffix)
        self.instance_images_path = [i for i in self.instance_images_path if i.suffix in ["png", "jpeg", "jpg"]]
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_images_"] = instance_image

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    images = [example["instance_images_"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "images": images}
    return batch


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


def main():
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    logging_dir = Path("logs", "logs")

    accelerator_project_config = ProjectConfiguration(project_dir="logs", logging_dir=logging_dir)

    

    model_root = '/home/dell/workspace/models/stable-diffusion-xl-base-1.0'
    output_dir = 'jlout'
    device = 'cuda'
    dtype = torch.float32
    train_batch_size = 1
    dataloader_num_workers = 2
    with_prior_preservation = False
    steps = 100
    num_train_epochs = steps
    gradient_accumulation_steps = 2
    prompt = 'jialin'
    resolution = 1024
    mixed_precision = "bf16"
    instance_data_dir = "/home/dell/workspace/xl_lora/jl_test"
    # instance_data_dir = "/home/dell/workspace/xl_lora/zdxpure"
    # image = Image.open('test/nq/nq.png')
    # image = image.resize((512,512), resample=Image.Resampling.LANCZOS)

    report_to = "wandb"
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
    )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        class_data_root=None,
        class_num=1,
        size=resolution,
        center_crop=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
        num_workers=dataloader_num_workers,
    )
    for it,it_ in enumerate(train_dataloader):
        1+1
        break

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
    
    # text_lora_parameters_1 = LoraLoaderMixin._modify_text_encoder(text_encoder_1, dtype=torch.float32)
    # text_lora_parameters_2 = LoraLoaderMixin._modify_text_encoder(text_encoder_2, dtype=torch.float32)
    
    optimizer = torch.optim.AdamW(
        # itertools.chain(unet_lora_parameters, text_lora_parameters_1, text_lora_parameters_2),
        itertools.chain(unet_lora_parameters),
        lr=0.0005
    )

    add_time_ids, _ = create_add_time_ids(
        (resolution, resolution),
        (0, 0),
        (resolution, resolution),
    )
    add_time_ids = add_time_ids.to(device, dtype=dtype)

    # init_image = create_np_image(image).to(device, dtype=dtype)
    # init_latent = vae.encode(init_image).latent_dist.sample()
    # init_latent = init_latent * vae.config.scaling_factor

    total_batch_size = train_batch_size * 2 * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {steps}")

    # pbar = tqdm(range(steps))
    global_step = 0
    pbar = tqdm(range(global_step, steps*2), disable=False)
    pbar.set_description('Steps')
    # for epoch in range(0, steps):
    for i in pbar: 
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            if 1:
                model_input = model_input.to(dtype)
            
            image = batch["images"]
            init_image = create_np_image(image).to(device, dtype=dtype)
            init_latent = vae.encode(init_image).latent_dist.sample()
            init_latent = init_latent * vae.config.scaling_factor
            # init_latent = model_input

            # Sample noise that we'll add to the latents
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
    # text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder_1.to(torch.float32))
    # text_encoder_2_lora_layers = text_encoder_lora_state_dict(text_encoder_2.to(torch.float32))
    
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_lora_layers,
        # text_encoder_lora_layers=text_encoder_lora_layers,
        # text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        text_encoder_lora_layers=None,
        text_encoder_2_lora_layers=None,
    )


if __name__ == "__main__":
    main()
