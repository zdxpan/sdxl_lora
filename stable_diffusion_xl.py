import copy
import diffusers
import inspect
import math, os, re, sys, time, random
import numpy as np
import torch
import torch.nn as nn

from contextlib                   import contextmanager
from diffusers                    import AutoencoderKL, UNet2DConditionModel
from PIL                          import Image, ImageOps
from safetensors.torch            import load_file
from transformers                 import CLIPConfig, CLIPVisionModel, PreTrainedModel
from transformers                 import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor
from tqdm                         import tqdm, trange

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)


SAMPLER_CHOICES = {
    'ddim':           diffusers.DDIMScheduler,
    'ddpm':           diffusers.DDPMScheduler,
    'deis':           diffusers.DEISMultistepScheduler,
    'k_dpmpp_2m':     diffusers.DPMSolverMultistepScheduler,
    'k_dpmpp_sde':    diffusers.DPMSolverSDEScheduler,
    'k_dpmpp_2s':     diffusers.DPMSolverSinglestepScheduler,
    'k_euler_a':      diffusers.EulerAncestralDiscreteScheduler,
    'k_euler':        diffusers.EulerDiscreteScheduler,
    'k_heun':         diffusers.HeunDiscreteScheduler,
    'k_dpm_2_a':      diffusers.KDPM2AncestralDiscreteScheduler,
    'k_dpm_2':        diffusers.KDPM2DiscreteScheduler,
    'lms':            diffusers.LMSDiscreteScheduler,
    'pndm':           diffusers.PNDMScheduler,
    'unipc':          diffusers.UniPCMultistepScheduler,
}


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def create_np_mask(image, factor=8):
    # convert into a black/white mask
    mask = Image.new(mode='L', size=image.size, color=255)
    mask.putdata(image.getdata(band=3))
    # print(
    #     f'>> DEBUG: writing the mask to mask.png'
    #     )
    # mask.save('mask.png')
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return mask


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


def create_random_tensors(shape, seed):
    x = []
    for s in seed:
        torch.manual_seed(s)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        x.append(torch.randn(shape, device='cpu'))
    return torch.cat(x)


def crop_image(image, width, height):
    w, h = image.size
    factor = min(w / width, h / height)
    width = int(width * factor)
    height = int(height * factor)
    image = image.crop((
        (w - width) // 2,
        (h - height) // 2,
        (w + width) // 2,
        (h + height) // 2
    ))
    return image


def create_tiles(latent, noise, max_size=64, min_size=64, overlap=8):
    height = latent.shape[2]
    width = latent.shape[3]
    result = [[],[]]

    for i, total in enumerate([height, width]):
        p = 0
        while True:
            remain = total - p
            if remain == 0:
                break
            elif remain < max_size:
                if remain < min_size:
                    result[i].append([p - (min_size - remain), min_size, overlap + (min_size - remain)])
                else:
                    result[i].append([p, remain, overlap])
                p += remain
            elif remain == max_size:
                result[i].append([p, max_size, overlap])
                p += max_size
            elif remain > max_size:
                result[i].append([p, max_size, overlap])
                p += max_size - overlap

    tiles = [[ None ] * len(result[1]) for i in range(len(result[0]))]

    for m, data_m in enumerate(result[0]):
        for n, data_n in enumerate(result[1]):
            y = data_m[0]
            x = data_n[0]
            h = data_m[1]
            w = data_n[1]
            tiles[m][n] = {
                'latent': latent[:,:,y:y+h,x:x+w],
                'noise': noise[:,:,y:y+h,x:x+w],
                'temp': None,
                'overlap': [
                    0 if m == 0 else data_m[2],
                    0 if n == 0 else data_n[2]
                ],
                'position': [ y, x ],
                'size': [ h, w ]
            }

    return tiles


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


def load_state_dict(pathname):
    print(f"Load state dict from {pathname}")
    if pathname.endswith('.safetensors'):
        pl_sd = load_file(pathname, device='cpu')
    else:
        pl_sd = torch.load(pathname, map_location='cpu')
    sd = pl_sd['state_dict'] if 'state_dict' in pl_sd else pl_sd
    return sd


def lora_mix(base_sd, lora_sd, lora_prefix, ratio, device='cpu', dtype=torch.float32):
    if device == 'cpu':
        dtype = torch.float32
    record_replace = {}
    for key, weight in base_sd.items():
        lora_key = match_lora_dict(key, lora_prefix)
        if lora_key and (lora_key + '.down.weight') in lora_sd:
            assert (lora_key + '.up.weight') in lora_sd

            down_weight = lora_sd.pop(lora_key + '.down.weight').to(device, dtype=dtype)
            up_weight = lora_sd.pop(lora_key + '.up.weight').to(device, dtype=dtype)

            if len(weight.size()) == 2:
                # linear
                delta = up_weight @ down_weight
            else:
                # conv2d
                # if (lora_key + '.lora_mid.weight') in lora_sd:
                #         mid_weight = lora_sd.pop(lora_key + '.lora_mid.weight').to(device, dtype=dtype)
                #         down_weight = merge_conv(mid_weight, down_weight.transpose(0,1)).transpose(0,1)
                #         del mid_weight
                delta = merge_conv(down_weight, up_weight)
            
            record_replace[key] = weight
            weight = (weight.to(device, dtype=dtype) + ratio * delta).to('cpu', dtype=dtype)
            del down_weight, up_weight, delta
            base_sd[key] = weight
    return base_sd, record_replace

def lora_de_mix(base_sd, record_replace, device='cpu', dtype=torch.float32):
    if device == 'cpu':
        dtype = torch.float32
    for key, weight in record_replace.items():
        base_sd[key] = weight
    return base_sd


def match_lora_dict(key, prefix):
    lora_key = None
    for target, replacement in [
        ('.to_q.weight', '.processor.to_q_lora'),
        ('.to_k.weight', '.processor.to_k_lora'),
        ('.to_v.weight', '.processor.to_v_lora'),
        ('.to_out.0.weight', '.processor.to_out_lora')
    ]:
        if key.endswith(target):
            lora_key = key.replace(target, replacement)
            lora_key = prefix + lora_key
            break
    return lora_key


def merge_conv(a, b):
    rank, in_ch, kernel_size, k_ = a.shape
    out_ch, rank_, _, _ = b.shape
    assert rank == rank_ and kernel_size == k_
    merged = b.reshape(out_ch, -1) @ a.reshape(rank, -1)
    merged = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    return merged


def normalize_size(width, height, max_size=2048):
    # the max area of an image should less than max_size * max_size
    coefficient = float(width * height) / (max_size * max_size)
    # resize
    if coefficient > 1:
        width, height = map(lambda x: int(x/math.sqrt(coefficient)), (width, height))
    # normalize to integer multiple of 8
    width, height = map(lambda x: max(math.ceil(x / 8) * 8, 512), (width, height))
    return width, height


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def pad_image(image, width, height):
    w, h = image.size
    target = Image.new(image.mode, (width, height))
    target.paste(image, (0, 0))
    line_v = target.crop((w - 1, 0, w, height))
    for x in range(w, width):
        target.paste(line_v, (x, 0))
    line_h = target.crop((0, h - 1, width, h))
    for y in range(h, height):
        target.paste(line_h, (0, y))
    return target


def resize_image(image, width, height):
    if image:
        w, h = image.size
        if w != width or h != height:
            image = crop_image(image, width, height)
            image = image.resize(
                (width, height),
                resample=Image.Resampling.LANCZOS
            )
            w, h = image.size
        if w % 8 != 0 or h % 8 != 0:
            image = pad_image(
                image,
                math.ceil(w / 8) * 8,
                math.ceil(h / 8) * 8
            )
    return image


def seed_to_int(seed, batch_size=1):
    seed = seed[:batch_size] + [-1] * (batch_size - len(seed))
    for i in range(len(seed)):
        if seed[i] == -1:
            random.seed()
            seed[i] = random.randint(0, 2**32 - 1)
    return seed


def slerp(t, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD=0.9995):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2)

    return v2


class StableDiffusionXL:
    @torch.no_grad()
    def __call__(self, opt={}):
        start_time = time.time()

        # 0. Lazy load
        if not self._initiated:
            print(f'>> Initiating StableDiffusionXL on the first request...')
            self.initiate()

        # 1. Prepare arguments
        opt_dict = vars(opt)
        for key in self.default_args.keys():
            if not key in opt_dict:
                opt_dict[key] = self.default_args[key]

        width, height = normalize_size(
            opt.width,
            opt.height,
            8192 if opt.tile is not None else 2048
        )
        do_classifier_free_guidance = opt.scale > 1.0

        # 2. Load models
        self.load_model(opt.lora)
        self.set_schedular(opt.sampler)
        factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        channels = self.vae.config.latent_channels

        model_loaded_time = time.time()

        # 3. Encode prompts
        with self.to_device(self.text_encoder, self.text_encoder_2):
            tokenizers = [self.tokenizer, self.tokenizer_2]
            text_encoders = [self.text_encoder, self.text_encoder_2]

            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_input = tokenizer(
                    [opt.prompt] * opt.n_samples,
                    padding='max_length',
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                prompt_embeds = text_encoder(
                    text_input.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            add_text_embeds = pooled_prompt_embeds

            if do_classifier_free_guidance:
                negative_prompt_embeds_list = []
                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    text_input = tokenizer(
                        [opt.neg_prompt] * opt.n_samples,
                        padding='max_length',
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors='pt'
                    )
                    negative_prompt_embeds = text_encoder(
                        text_input.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                    negative_prompt_embeds_list.append(negative_prompt_embeds)
                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)

        # 4. Prepare init noise
        seed = seed_to_int(opt.seed, opt.n_samples)
        base_x = create_random_tensors(
            [1, channels, height // factor, width // factor],
            seed
        )
        if opt.image is not None and opt.strength < 0: # repaint (obsolete)
            threshold = 0.5 + (abs(opt.strength) * 2)**2 / 2
            base_x = torch.clamp(base_x, min=-threshold, max=threshold)
        if opt.variant == 0.0:
            variant_seed = [0] * opt.n_samples
            init_noise = base_x
        else:
            variant_seed = seed_to_int(opt.variant_seed, opt.n_samples)
            variant_x = create_random_tensors(
                [1, channels, height // factor, width // factor],
                variant_seed
            )
            init_noise = slerp(
                max(0.0, min(1.0, opt.variant)),
                base_x,
                variant_x
            )
        init_noise = init_noise.to(self.device, dtype=self.dtype)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(opt.steps, device=self.device)
        steps = opt.steps
        timesteps = self.scheduler.timesteps
        relay_point = (
            timesteps[int(len(timesteps) * (1.0 - opt.refine))]
        ) if opt.refine > 0 else -1

        # 6. Prepare init image and mask
        init_mask = None
        latent_mask = None
        if opt.mask is not None:
            init_mask = resize_image(opt.file[opt.mask], opt.width, opt.height)
            init_mask = create_np_mask(init_mask, factor)
            init_mask = init_mask.to(self.device, dtype=self.dtype)

            latent_mask = torch.nn.functional.interpolate(
                init_mask[:, 0:1, :, :],
                size=(height // factor, width // factor)
            ).repeat(1, 4, 1, 1)
            latent_mask[latent_mask < 0.5] = 0
            latent_mask[latent_mask >= 0.5] = 1

        init_image = None
        image_condition = None
        if opt.image is not None:
            with self.to_device(self.vae):
                steps = min(int(opt.steps * abs(opt.strength)), opt.steps)
                start_step = max(opt.steps - steps, 0)
                timesteps = timesteps[start_step * self.scheduler.order:]

                init_image = resize_image(opt.file[opt.image], opt.width, opt.height)
                init_image = create_np_image(init_image)
                init_image = init_image.to(self.device, dtype=self.dtype)

                init_latent = self.vae.encode(init_image).latent_dist.sample()
                init_latent = init_latent * self.vae.config.scaling_factor

                if opt.boost != 0.0:
                    print('>> Use latent image upscaler')
                    tmp_latent = torch.nn.functional.interpolate(
                        init_latent,
                        scale_factor=0.5,
                        mode='bilinear'
                    )
                    tmp_latent = torch.nn.functional.interpolate(
                        tmp_latent,
                        size=(height // factor, width // factor),
                        mode='bilinear'
                    )
                    tmp_latent = init_latent * (1. - opt.boost) + tmp_latent * opt.boost

                    if latent_mask is not None:
                        init_latent = init_latent * latent_mask + (1. - latent_mask) * tmp_latent
                    else:
                        init_latent = tmp_latent

                init_latent = init_latent.repeat(opt.n_samples, 1, 1, 1)

                if abs(opt.strength) == 1.0:
                    # Even strength equals to 1.0, the output of scheduler.add_noise
                    # does not equal to init_noise, this may affect inpaiting result
                    # especially when the input image has a pure background. So we
                    # set the latent to init_noise to completely ignore the input image.
                    latent = init_noise * self.scheduler.init_noise_sigma
                else:
                    latent = self.scheduler.add_noise(
                        init_latent,
                        init_noise,
                        timesteps[:1].repeat(opt.n_samples)
                    ) if len(timesteps) > 0 else init_latent
        else:
            init_latent = init_noise * self.scheduler.init_noise_sigma
            latent = init_latent

        if latent_mask is not None:
            latent_mask = latent_mask.repeat(opt.n_samples, 1, 1, 1)

        # 7. Generate controlnet condition image
        # TODO

        # 8. Denoising loop
        # Use tiled UNet to process large latent
        # Because it's relatively hard to ensure the consistance between tiles,
        # try to increase max tile size to reduce the tile numbers
        full_latent = latent
        full_noise_pred = torch.zeros_like(latent, device=self.device, dtype=self.dtype)
        full_noise_pred = torch.cat(
            [ full_noise_pred ] * 2
        ) if do_classifier_free_guidance else full_noise_pred
        tiles = create_tiles(
            full_latent,
            full_noise_pred,
            max_size=(1280 if opt.tile is None else opt.tile[0]) // factor,
            min_size=768 // factor,
            overlap=(64 if opt.tile is None else opt.tile[1]) // factor,
        )
        n_y = len(tiles)
        n_x = len(tiles[0])
        if n_y * n_x > 1:
            print(f'>> Use tiled UNet with {n_y}x{n_x} tiles')

        # Start!
        self.use_unet('base')
        stage = 1
        add_time_ids_cache = {}
        num_warmup_steps = len(timesteps) - steps * self.scheduler.order
        with tqdm(total=steps) as pbar:
            for i, t in enumerate(timesteps):
                if stage == 1 and t <= relay_point:
                    self.use_unet('refiner')
                    stage = 2
                    # unet_refiner use only text_encoder_2
                    hidden_size = self.text_encoder_2.config.hidden_size
                    prompt_embeds = prompt_embeds[:,:,-hidden_size:]

                for m in range(n_y):
                    for n in range(n_x):
                        # Get current tile and corresponding scheduler
                        tile = tiles[m][n]
                        latent = tile['latent']
                        y = tile['position'][0]
                        x = tile['position'][1]
                        h = tile['size'][0]
                        w = tile['size'][1]

                        # Create add_time_ids using current tile's position and size.
                        # This may reduce the dumplicate concepts in each tile.
                        if not (stage, m, n) in add_time_ids_cache:
                            if stage == 2:
                                add_time_ids, add_neg_time_ids = create_add_time_ids(
                                    (height, width),
                                    (y * factor, x * factor),
                                    (h * factor, w * factor),
                                    requires_aesthetics_score=True
                                )
                                add_time_ids = add_time_ids.to(self.device, dtype=self.dtype)
                                add_neg_time_ids = add_neg_time_ids.to(self.device, dtype=self.dtype)
                                if do_classifier_free_guidance:
                                    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
                            else:
                                add_time_ids, _ = create_add_time_ids(
                                    (height, width),
                                    (y * factor, x * factor),
                                    (h * factor, w * factor)
                                )
                                add_time_ids = add_time_ids.to(self.device, dtype=self.dtype)
                                if do_classifier_free_guidance:
                                    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
                            add_time_ids_cache[(stage, m, n)] = add_time_ids.repeat(opt.n_samples, 1)
                        add_time_ids = add_time_ids_cache[(stage, m, n)]

                        # expand the latents if we are doing classifier free guidance
                        # The latents are expanded 3 times because for pix2pix the guidance\
                        # is applied for both the text and the input image.
                        latent_model_input = torch.cat(
                            [latent] * 2
                        ) if do_classifier_free_guidance else latent

                        # concat latents, image_condition in the channel dimension
                        scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        if image_condition is not None:
                            scaled_latent_model_input = torch.cat(
                                [scaled_latent_model_input, image_condition[:,:,y:y+h,x:x+w]],
                                dim=1
                            )

                        # predict the noise residual
                        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                        noise_pred = self.unet(
                            scaled_latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False
                        )[0]

                        # predicted tiled noise will be merged in the outer loop
                        tile['temp'] = noise_pred

                # merge all predicted noise
                for m in range(n_y):
                    for n in range(n_x):
                        tile = tiles[m][n]
                        noise_pred = tile['temp']
                        o_h = tile['overlap'][0]
                        o_w = tile['overlap'][1]
                        for j in range(o_h):
                            ratio = j / o_h
                            noise_pred[:,:,j:j+1,:] = noise_pred[:,:,j:j+1,:] * ratio + tile['noise'][:,:,j:j+1,:] * (1 - ratio)
                        for j in range(o_w):
                            ratio = j / o_w
                            noise_pred[:,:,:,j:j+1] = noise_pred[:,:,:,j:j+1] * ratio + tile['noise'][:,:,:,j:j+1] * (1 - ratio)
                        tile['noise'][:,:,:,:] = noise_pred

                noise_pred = full_noise_pred

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + opt.scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and opt.rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=opt.rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latent = self.scheduler.step(
                    noise_pred,
                    t,
                    # The value of full_latent will be updated later,
                    # create a copy to avoid break the inner state of some schedulers
                    torch.clone(full_latent),
                    return_dict=False
                )[0]

                # apply latent mask
                if latent_mask is not None:
                    init_latent_proper = self.scheduler.add_noise(
                        init_latent,
                        noise_pred_uncond if do_classifier_free_guidance else init_noise,
                        timesteps[i + 1:i + 2].repeat(opt.n_samples)
                    ) if i < len(timesteps) - 1 else init_latent
                    latent = init_latent_proper * latent_mask + (1. - latent_mask) * latent

                # update the full latent
                full_latent[:,:,:,:] = latent

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    pbar.update()

        latent = full_latent
        self.free_unet()

        # 9. Decode latent
        with self.to_device(self.vae):
            latent = (1 / self.vae.config.scaling_factor) * latent
            x_samples = self.vae.decode(latent).sample
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
            x_samples = x_samples[:,:opt.height,:opt.width,:]
            samples = [
                Image.fromarray(x)
                for x in (x_samples * 255).round().astype(np.uint8)
            ]

        inference_complete_time = time.time()

        # 10. Safety check
        has_nsfw_concept = self.check_safety(x_samples, samples)

        for i in range(len(samples)):
            sample = samples[i]
            # Check black image. Sometimes FP16 mode generates black images.
            if all([sum(sample.getextrema()[channel]) == 0 for channel in range(3)]):
                print(f'Black output image[{i}] detected.')
                # Pretend to be a NSFW content,
                # so the user could retry with different paremeters.
                has_nsfw_concept[i] = True

        print(f'>> Model loading costs {model_loaded_time - start_time:.2f}s, '
            f'inference costs {inference_complete_time - model_loaded_time:.2f}s, '
            f'safety checking costs {time.time() - inference_complete_time:.2f}s')

        return {
            'images': samples,
            'nsfw': has_nsfw_concept,
            'prompt': opt.prompt,
            'neg_prompt': opt.neg_prompt,
            'seed': seed,
            'variant_seed': variant_seed,
            'variant': opt.variant,
            'batch_size': opt.n_samples,
            'steps': opt.steps,
            'scale': opt.scale,
            'sampler': opt.sampler,
            'clip_skip': 2,
            'model': 'stable-diffusion-xl',
            'vae': 'sdxl-vae',
            'lora': opt.lora,
            'control': []
        }

    @torch.no_grad()
    def __init__(self, base, index, device, dtype=torch.float16):
        self.base = base
        self.dict = index
        self.device = device
        self.dtype = dtype

        self._initiated = False
        self._last_used_unet = None
        self._last_mixed_lora = []

        self.default_args = {
            'lora': [],
            'sampler': 'ddim',
            'steps': 30,
            'width': 512,
            'height': 512,
            'files': [],
            'image': None,
            'mask': None,
            'boost': 0.0,
            'strength': 0.75,
            'seed': [],
            'prompt': '',
            'neg_prompt': '',
            'n_samples': 1,
            'variant': 0.0,
            'variant_seed': [],
            'scale': 7.5,
            'rescale': 0.7,
            'refine': 0.3,
            'tile': None
        }

    def apply_optimization(self):
        if str(self.device) == 'cuda':
            print('>> Enbalbe xformers')
            self.unet_base.enable_xformers_memory_efficient_attention()
            self.unet_refiner.enable_xformers_memory_efficient_attention()
            print('>> Enbalbe VAE tiling')
            self.vae.enable_tiling()
            print('>> Enbalbe VAE slicing')
            self.vae.enable_slicing()

    def check_safety(self, x_image, image):
        with self.to_device(self.safety_checker):
            safety_checker_input = self.feature_extractor(
                image,
                return_tensors='pt'
            )
            has_nsfw_concept = self.safety_checker(
                safety_checker_input.pixel_values.to(self.device, self.dtype)
            )
            for i in range(len(has_nsfw_concept)):
                if has_nsfw_concept[i]:
                    print(f'>> NSFW output image[{i}] detected.')
        return has_nsfw_concept

    def exist(self, lora=None):
        if lora and not lora in self.dict['lora']:
            return False
        return True

    def free_unet(self):
        if self.unet != None:
            self.unet.to('cpu')
            self.unet = None
            self._last_used_unet = None

    def get_lora_names(self):
        return list(self.dict['lora'].keys())

    def get_schedular_names(self):
        return list(SAMPLER_CHOICES)

    def initiate(self):
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.base['v1'],
            subfolder='feature_extractor',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            self.base['v1'],
            subfolder='safety_checker_x',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.scheduler = diffusers.EulerDiscreteScheduler.from_pretrained(
            self.base['base'],
            subfolder='scheduler',
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.unet_base = UNet2DConditionModel.from_pretrained(
            self.base['base'],
            subfolder='unet',
            torch_dtype = self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.unet_refiner = UNet2DConditionModel.from_pretrained(
            self.base['refiner'],
            subfolder='unet',
            torch_dtype = self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base['base'],
            subfolder='text_encoder',
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.base['base'],
            subfolder='text_encoder_2',
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base['base'],
            subfolder='tokenizer',
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.base['base'],
            subfolder='tokenizer_2',
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.base['base'],
            subfolder='vae_fp16_fix',
            torch_dtype = self.dtype,
            use_safetensors=True,
            variant='fp16'
        )

        self.unet = None

        self.apply_optimization()
        self._initiated = True

    def load_model(self, lora=[]):
        lora = list(map(lambda x: (self.dict['lora'][x[0]], x[1]), lora))

        if self._last_mixed_lora != lora:
            unet_base_sd = load_state_dict(os.path.join(self.base['base'], 'unet/diffusion_pytorch_model.fp16.safetensors'))
            unet_refiner_sd = load_state_dict(os.path.join(self.base['refiner'], 'unet/diffusion_pytorch_model.fp16.safetensors'))
            for pathname, ratio in lora:
                lora_sd = load_state_dict(pathname)
                if pathname.endswith('refiner.bin'):
                    unet_sd = unet_refiner_sd
                else:
                    unet_sd = unet_base_sd
                lora_mix(unet_sd, lora_sd, 'unet.', ratio, self.device, self.dtype)
                if len(lora_sd) > 0:
                    print(">> Unexpected LoRA keys:")
                    print(lora_sd.keys())
            self.unet_base.load_state_dict(unet_base_sd)
            self.unet_refiner.load_state_dict(unet_refiner_sd)
            self._last_mixed_lora = lora

    def set_schedular(self, name):
        print(f'>> Setting scheduler to {name}')
        self.scheduler = SAMPLER_CHOICES[name].from_config(
            self.scheduler.config
        )

    @contextmanager
    def to_device(self, *models):
        for model in models:
            model.to(self.device)
        try:
            yield None
        finally:
            for model in models:
                model.to('cpu')

    def use_unet(self, model):
        if self._last_used_unet != model:
            if self.unet is not None:
                self.unet.to('cpu')
            if model == 'base':
                self.unet = self.unet_base
            elif model == 'refiner':
                self.unet = self.unet_refiner
            else:
                raise Exception(f'Unsupported UNet type {model}')
            self.unet.to(self.device)
            self._last_used_unet = model


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

        self.bn = nn.BatchNorm1d(17, eps=0.001, momentum=0.1, affine=True)
        self.classifier = nn.Linear(config.projection_dim, 17)
        self.classifier2 = nn.Linear(37, 1)
        self.last_bn = nn.BatchNorm1d(37, eps=0.001, momentum=0.1, affine=True)
        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist_ = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist_ = cosine_distance(image_embeds, self.concept_embeds)
        
        special_cos_dist = special_cos_dist_.cpu().float().numpy()
        cos_dist = cos_dist_.cpu().float().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0
            bad_concepts_threshold = 0
            nsfw_binary_threshold = 0.5

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > bad_concepts_threshold for res in result]

        # Also identify nsfw contents by custom checker
        out = self.classifier(image_embeds)
        out = self.bn(out)
        out = torch.concat(
            (special_cos_dist_, cos_dist_, out),
            dim=1
        ) # [batch, 37]
        out = self.last_bn(out)
        out = self.classifier2(out)
        out = self.sigmoid(out)
        out = out.squeeze(dim=1)

        has_nsfw_concepts_x = (out > nsfw_binary_threshold).cpu().numpy().tolist()

        # Vote the final result
        return [
            result and result_x
            for result, result_x in zip(has_nsfw_concepts, has_nsfw_concepts_x)
        ]
