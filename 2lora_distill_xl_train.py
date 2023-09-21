# @title Main code segment. Extremely long! Make sure you run it though.
import argparse
import functools
import itertools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict
import copy

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
# from t2i import Adapter

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    #T2IAdapter,
)
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
# from pipeline_xl_adapter import StableDiffusionXLAdapterPipeline
from prodigyopt import Prodigy

if is_wandb_available():
    import wandb
from torch.optim import Optimizer
from torchvision import transforms as T
from copy import deepcopy

torch.backends.cudnn.benchmark = True
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def is_loss_nan(loss):
    return torch.isnan(loss).any()

def to_device(*models):
    for model in models:
        model.to("cuda")
    try:
        yield None
    finally:
        for model in models:
            model.to('cpu')

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(
    vae, unet, args, accelerator, weight_dtype, step, test_dataloader
    # vae, unet, text_encoder, tokenizer, args, accelerator, weight_dtype, epoch
):
    # Perform validation loss calculations
    logger.info(">>  TODO  add lora Running validation... ")


    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        variant="fp16",
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # if args.enable_xformers_memory_efficient_attention:
    #     pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    print(">> all test data ", len(test_dataloader))
    image_logs = []
    images = []
    # for i in range(len(args.validation_prompts)):
    for x in test_dataloader:
        validation_prompt = x[args.caption_column]
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                # image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                image = pipeline(validation_prompt, 
                                num_inference_steps=20, 
                                generator=generator, 
                                height=args.resolution,
                                width=args.resolution,
                                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "images": images,
                "validation_prompt": validation_prompt,
            }
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                # control_image = log["validation_image"]

                formatted_images = []

                # formatted_images.append(np.asarray(control_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                # control_image = log["validation_image"]

                # formatted_images.append(
                #     wandb.Image(control_image, caption="t2i_adapter conditioning")
                # )

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- diffusers
- t2i_adapter
inference: true
---
    """
    model_card = f"""
# lora_distill-{repo_id}

These arelora_distill weights trained on {base_model}
{img_str}
"""
    model_card += """

## License

[SDXL 1.0 License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="lora distill training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=(
            "Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=float,
        default=0.5,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_prodigy",
        action="store_true",
        help="Whether or not to use prodigy optimizer from Meta labs.",
    )
    parser.add_argument(
        "--use_adamcm",
        action="store_true",
        help="Whether or not to use Adam with critical momenta.",
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
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image.",
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="guide",
        help="The column of the dataset containing the t2i_adapter conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the t2i_adapter conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_t2i_adapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is None:
        raise ValueError(
            "`--validation_image` must be set if `--validation_prompt` is set"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the t2i_adapter encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    print(">> -dataset_name in load", args.dataset_name)
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    with accelerator.main_process_first():
        train_dataset = dataset.shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

    return train_dataset, test_dataset


def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
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


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True
):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def prepare_train_dataset(dataset, accelerator):
    p = 0.05
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def preprocess_train(examples):
        #augment = transforms.TrivialAugmentWide()
        # images = [image.convert("RGB") for image in examples[args.image_column]]
        # images = [image_transforms(image) for image in images]
        # examples["pixel_values"] = images
        examples["caption"] = [i for i in examples["text"]]
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset

def prepare_test_dataset(dataset, accelerator):
    p = 0.05
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )
    def preprocess_test(examples):
        # images = [image.convert("RGB") for image in examples[args.image_column]]
        # images = [image_transforms(image) for image in images]
        # examples["pixel_values"] = images
        # examples["conditioning_pixel_values"] = conditioning_images
        examples["caption"] = [i for i in examples["text"]]
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_test)

    return dataset

def train_collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # conditioning_pixel_values = torch.stack(
    #     [example["conditioning_pixel_values"] for example in examples]
    # )
    # conditioning_pixel_values = conditioning_pixel_values.to(
    #     memory_format=torch.contiguous_format
    # ).float()
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompt_ids = torch.stack(
        [torch.tensor(example["prompt_embeds"]) for example in examples]
    )

    add_text_embeds = torch.stack(
        [torch.tensor(example["text_embeds"]) for example in examples]
    )
    add_time_ids = torch.stack(
        [torch.tensor(example["time_ids"]) for example in examples]
    )

    return {
        # "pixel_values": pixel_values,
        # "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        },
    }

def test_collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # conditioning_pixel_values = torch.stack(
    #     [example["conditioning_pixel_values"] for example in examples]
    # )
    # conditioning_pixel_values = conditioning_pixel_values.to(
    #     memory_format=torch.contiguous_format
    # ).float()
    # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    return {
        # "pixel_values": pixel_values,
        # "conditioning_pixel_values": conditioning_pixel_values,
        "caption": [example["caption"] for example in examples],
        "text": [example["caption"] for example in examples],
    }

def create_random_tensors(shape, seed):
    x = []
    for s in seed:
        torch.manual_seed(s)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        x.append(torch.randn(shape, device='cpu'))
    return torch.cat(x)

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    print(">>> ", args.pretrained_model_name_or_path)
    # text_encoder_one = text_encoder_cls_one.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     # subfolder="text_encoder",
    #     # revision=args.revision,
    # )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder='text_encoder',
        torch_dtype=args.dtype,
        revision=args.revision,
        use_safetensors=True,
        variant='fp16'
    )
    text_encoder_one_db = text_encoder_cls_one.from_pretrained(
        args.db_teacher_model_name_or_path,
        subfolder='text_encoder',
        torch_dtype=args.dtype,
        revision=args.revision,
        use_safetensors=True,
        variant='fp16'
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder='text_encoder_2',
            torch_dtype=args.dtype,
            use_safetensors=True,
            variant='fp16'
        )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder='vae_fp16_fix',
        torch_dtype = args.dtype,
        use_safetensors=True,
        variant='fp16'
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        #     subfolder="unet", revision=args.revision
        # )
        subfolder='unet',
        torch_dtype = args.dtype,
        use_safetensors=True,
        variant='fp16'
    )
    unet_teacher_kd = UNet2DConditionModel.from_pretrained(
        args.db_teacher_model_name_or_path, 
        subfolder='unet',
        torch_dtype = args.dtype,
        use_safetensors=True,
        variant='fp16'
    )
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    unet_teacher_kd.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_one_db.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    print(f">> dtype of vae:{vae.dtype}  uent:{unet.dtype}, text_encoder_one:{text_encoder_one.dtype} \
          text_encoder_two:{text_encoder_two.dtype}  unet_teacher_kd{unet_teacher_kd.dtype}")
    
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)
    unet_teacher_kd.to(accelerator.device, dtype=weight_dtype)
    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # -------- now we will add new LoRA weights to the attention layers -------------------
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
        module_ = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.rank
        )
        unet_lora_attn_procs[name] = module_
        unet_lora_parameters.extend(module_.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)
    # Set correct lora layers
    # pipexl.load_lora_weights(lora_sd)
    
    # lora_layers = AttnProcsLayers(unet.attn_processors)
    
    # def compute_snr(timesteps):
    #     """
    #     Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    #     """
    #     alphas_cumprod = noise_scheduler.alphas_cumprod
    #     sqrt_alphas_cumprod = alphas_cumprod**0.5
    #     sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    #     # Expand the tensors.
    #     # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    #     sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    #     while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
    #         sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    #     alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    #     sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    #     while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
    #         sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    #     sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    #     # Compute SNR.
    #     snr = (alpha / sigma) ** 2
    #     return snr

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
            text_encoder_one, dtype=torch.float32, rank=args.rank
        )
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
            text_encoder_two, dtype=torch.float32, rank=args.rank
        )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(model)
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

        text_encoder_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_one_
        )

        text_encoder_2_state_dict = {k: v for k, v in lora_state_dict.items() if "text_encoder_2." in k}
        LoraLoaderMixin.load_lora_into_text_encoder(
            text_encoder_2_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_two_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if accelerator.unwrap_model(t2i).dtype != torch.float32:
    #     raise ValueError(
    #         f"t2i_adapter loaded as datatype {accelerator.unwrap_model(t2i).dtype}. {low_precision_error_string}"
    #     )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    print(">> lreaning rate:", args.learning_rate )
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    elif args.use_adamcm:
        optimizer_class = lambda params, lr, **b: AdamCM(params, lr=lr)
    else:
        optimizer_class = torch.optim.AdamW

    if args.use_prodigy:
        optimizer_class = lambda params, *a, **b: Prodigy(
            params
        )  # Prodigy has *some* trainable parameters, but they aren't particularly relevant to use in this context.

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
        if args.train_text_encoder
        else unet_lora_parameters
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    unet, text_encoder_one, text_encoder_two, vae = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, vae
    )

    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(
        batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True
    ):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (0, 0)
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        unet_added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    train_dataset, test_dataset = get_train_dataset(args, accelerator)
    train_dataset = train_dataset["train"]
    print(f">> data_set len:", len(train_dataset))
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(
            compute_embeddings_fn,
            batched=True,
            batch_size=4,
            new_fingerprint=new_fingerprint,
        )
    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    # Then get the training dataset ready to be passed to the dataloader.
    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    test_dataset = prepare_test_dataset(test_dataset, accelerator)
    print(">> kelen test_dataset", len(test_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_collate_fn,
        batch_size=args.train_batch_size,
        # num_workers=args.dataloader_num_workers,
        num_workers=4,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        # "cosine",
        args.lr_scheduler,
        optimizer=optimizer,
        # num_warmup_steps=0,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        optimizer, train_dataloader, unet, text_encoder_one, text_encoder_two, lr_scheduler, unet_teacher_kd     = accelerator.prepare(
            optimizer, train_dataloader, unet, text_encoder_one, text_encoder_two, lr_scheduler, unet_teacher_kd
        )
    else:
        optimizer, train_dataloader, unet, text_encoder_one, text_encoder_two, lr_scheduler, unet_teacher_kd = accelerator.prepare(
            optimizer, train_dataloader, unet, text_encoder_one, text_encoder_two, lr_scheduler, unet_teacher_kd
        )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        # tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # t2i.load_state(os.path.join(args.output_dir, path))
            # TODO load lora state_dict
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    if args.steps_ep > 0:
        initial_global_step = args.steps_ep
        global_step = args.steps_ep

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    # --  在每个集货层进行蒸馏损失对齐，让其在这些层学习目标一致~-------
    # KD_teacher = {}
    # KD_student= {}
    # num_blocks= 4 if args.distill_level=="sd_small" else 3
    # def getActivation(activation,name,residuals_present):
    #     # the hook signature
    #     if residuals_present:
    #         def hook(model, input, output):
    #             activation[name] = output[0]
    #     else:
    #         def hook(model, input, output):
    #             activation[name] = output
    #     return hook
        
    # def cast_hook(unet,dicts,model_type,teacher=False):
    #     unet=accelerator.unwrap_model(unet)
    #     if teacher:   # -- 由于不修改模型结构，因此这一步就足够使用了， 推算时候需要调用该🪝处理
    #         for i in range(4):
    #             unet.down_blocks[i].register_forward_hook(getActivation(dicts,'d'+str(i),True))
    #         unet.mid_block.register_forward_hook(getActivation(dicts,'m',False))
    #         for i in range(4):
    #             unet.up_blocks[i].register_forward_hook(getActivation(dicts,'u'+str(i),False))
    #     else:
    #         num_blocks= 4 if model_type=="sd_small" else 3
    #         for i in range(num_blocks):
    #             unet.down_blocks[i].register_forward_hook(getActivation(dicts,'d'+str(i),True))
    #         if model_type=="sd_small":
    #             unet.mid_block.register_forward_hook(getActivation(dicts,'m',False))
    #         for i in range(num_blocks):
    #             unet.up_blocks[i].register_forward_hook(getActivation(dicts,'u'+str(i),False))
    
    # cast_hook(unet,KD_student, args.distill_level,False)
    # cast_hook(unet_teacher_kd, KD_teacher,args.distill_level,True)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()   # TODO  ,  上面已经设置unet grad False , 现在还ntrai是不是会导致整个模型都会被训练呢？
        if args.train_text_encoder:
            text_encoder_one.train()
            # text_encoder_two.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if args.use_image_train:
                    pixel_values = batch["pixel_values"]
                    # pixel_values = pixel_values.to(args.dtype)
                    latents = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                else:
                    channels = vae.config.latent_channels
                    factor = 2 ** (len(vae.config.block_out_channels) - 1)
                    bsz, _, _ = batch["prompt_ids"].shape
                    init_latent = create_random_tensors(
                        [bsz, channels, args.resolution // factor, args.resolution // factor],
                           [args.seed]
                        )
                    init_latent = init_latent * noise_scheduler.init_noise_sigma
                    init_latent = init_latent.to(unet.device, dtype=unet.dtype)
                    latents = init_latent

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                # if args.input_perturbation:   same as up code snipte
                #     new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # if args.input_perturbation:
                #     noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                # else
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # lora_adapter conditioning.  TODO
                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                    
                # Predict the noise residual
                with torch.no_grad():
                    teacher_pred=unet_teacher_kd(noisy_latents, 
                                                timesteps, 
                                                encoder_hidden_states=batch["prompt_ids"],
                                                added_cond_kwargs=batch["unet_added_conditions"],
                                ).sample
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    # down_block_additional_residuals=down_block_res_samples,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                # -- distill make layer mathch loss ~~~
                # loss_features=0
                # for i in range(4):
                #     loss_features=loss_features+F.mse_loss(KD_teacher['d'+str(i)],KD_student['d'+str(i)])
                # loss_features=loss_features+F.mse_loss(KD_teacher['m'],KD_student['m'])
                # for i in range(4):
                #     loss_features=loss_features+F.mse_loss(KD_teacher['u'+str(i)],KD_student['u'+str(i)])

                loss_KD=F.mse_loss(model_pred.float(), teacher_pred.float(), reduction="mean")
                
                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
#                     传统上，扩散模型通过去噪一系列噪声样本来生成高质量的图像。目标是从最终的噪声样本中预测出原始的清晰图像。然而，在这种方法中，模型不直接预测原始的清晰图像，而是预测噪声本身。

# 为了适应这种修改后的公式，作者在同一篇论文的第4.2节中介绍了一种策略。该策略涉及根据每个扩散过程中的截断信噪比（SNR）计算损失权重。

# SNR是衡量信号与噪声质量的指标。通过对SNR进行截断，可以有效平衡优化过程中各个时间步之间的冲突。这意味着损失权重根据SNR进行调整，更重视具有较高SNR的时间步，从而实现更高效的训练。

# 关于如何计算损失权重以及如何利用截断的SNR平衡时间步之间的冲突的具体细节可以在论文的第4.2节中找到。
                    snr = compute_snr(timesteps, noise_scheduler)
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()
                    # loss = loss * 5 # 继续放大，太难收敛了~
            
                # Gather the losses across all processes for logging (if we use distributed training).
                loss = loss + args.output_weight*loss_KD   #  +args.feature_weight*loss_features
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one )  #, text_lora_parameters_two)
                        if args.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                # checkpoint1  t2i parameters state_dict  grad grad_fn -----------------------
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        # accelerator.save_state(save_path)
                        logger.info(f" >> TODO reimpliment Saved state to {save_path}")
                        # TODO  save the lora delta 
                        # t2i = accelerator.unwrap_model(t2i)
                        # t2i.save_pretrained(save_path)


                    if global_step % args.validation_steps == 0 or global_step == 1:
                        image_logs = log_validation(
                            vae,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            step,
                            test_dataloader,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder_one)
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = text_encoder_lora_state_dict(text_encoder_two)
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

        # Final inference
        # Load previous pipeline
        # pipeline = StableDiffusionXLPipeline.from_pretrained(
        #     args.pretrained_model_name_or_path, vae=vae, revision=args.revision, torch_dtype=weight_dtype
        # )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                # unet=unet,
                torch_dtype=weight_dtype,
                use_safetensors=True,
                variant="fp16",
            )

        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


# @title
if __name__ == "__main__":
    os.chdir("/home/dell/workspace/xl_lora/")
    # args = parse_args()
    from argparse import Namespace

    args = Namespace(
        pretrained_model_name_or_path="/home/dell/workspace/models/stable-diffusion-xl-base-1.0/",
        db_teacher_model_name_or_path = "/home/dell/workspace/models/finalAnimeCG_mk2a2/",
        target_lora_model_name_path="/home/dell/workspace/models/pixel-art-xl.safetensors",  # use unet.set lora attanprocess  is delta of base
        pretrained_vae_model_name_or_path="/home/dell/workspace/models/stable-diffusion-xl-base-1.0/vae_fp16_fix/",
        steps_ep=0,
        revision=None,
        output_dir="./outputs",
        cache_dir=None,
        snr_gamma=5.0,
        output_weight = 1,
        feature_weight = 1,
        local_rank = 128,
        validation_epochs = 5,  # ="Run validation every X epochs.",
        train_text_encoder = True,
        use_image_train = False,  #  不使用图片进行训练，蒸馏时只需要学会对应词
        # input_perturbation = False,   # 噪声加一个偏移，从而训练的模型能够生成更明亮的图片
        noise_offset = 0,            # 噪声加一个偏移，从而训练的模型能够生成更明亮的图片
        rank = 128,   # lora rank 128
        prediction_type = None,
        # snr_gamma=None,
        seed=5,
        resolution=768,
        crops_coords_top_left_h=0,
        crops_coords_top_left_w=0,
        train_batch_size=2,  # 2xf16: 26G  29G，哎  4xf16:  33G   8xf16: 48G
        num_train_epochs=10,
        dtype=torch.float16,
        # dtype=torch.float32,
        max_train_steps=None,
        checkpointing_steps=2000,     #  200 步保存一次检查点 如果检查点特别大，那就 500次保存一次  200 
        checkpoints_total_limit=5,
        resume_from_checkpoint=None,
        # resume_from_checkpoint="latest",    # bad not ok
        gradient_accumulation_steps=1,     # 可以增大， 相当于提升batch_size
        gradient_checkpointing=True,
        learning_rate=1.0,
        scale_lr=False,
        lr_scheduler="cosine",
        lr_warmup_steps=0,
        # lr_num_cycles=0.5,
        lr_num_cycles=1,
        lr_power=1.0,
        use_8bit_adam=False,
        # use_prodigy=False,
        use_prodigy=True,
        use_adamcm=True,
        dataloader_num_workers=4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=0.001,   # 0,01  default
        adam_epsilon=1e-08,
        max_grad_norm=1.0,
        push_to_hub=False,
        hub_token=None,
        hub_model_id="lora_distill_db",
        logging_dir="logs",
        allow_tf32=True,   # if it will no grad?
        report_to="wandb",
        mixed_precision="bf16",     # if it will no update?
        enable_xformers_memory_efficient_attention=True,  
        set_grads_to_none=True,
        dataset_name="./fill50kx/fill50kx.py",
        dataset_config_name=None,
        train_data_dir=None,
        image_column="image",
        conditioning_image_column="conditioning_image",
        caption_column="text",
        max_train_samples=None,
        proportion_empty_prompts=0.1,
        validation_prompt=None,
        validation_image=None,
        num_validation_images=4,
        validation_steps=2000,
        tracker_project_name="sd_xl_finetune_db_lora",
    )
    # print(args)
    main(args)


# Configuration saved in ./base/config.json
# Model weights saved in ./base/diffusion_pytorch_model.bin