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
from transformers                 import CLIPConfig, CLIPVisionModel, PreTrainedModel
from transformers                 import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from safetensors.torch            import load_file
from tqdm                         import tqdm, trange


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


def controlnet_mix(base_sd, unet_sd):
    for key in base_sd.keys():
        if key in unet_sd:
            if key == 'conv_in.weight':
                base_sd[key] += unet_sd[key][:, 0:4, :, :]
                base_sd[key] = torch.cat(
                    [ base_sd[key], unet_sd[key][:, 4:, :, :] ],
                    dim=1
                )
            else:
                base_sd[key] += unet_sd[key]


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
    for key, weight in base_sd.items():
        lora_key = match_lora_dict(key, lora_prefix)
        if lora_key and (lora_key + '.alpha') in lora_sd:
            alpha = lora_sd.pop(lora_key + '.alpha').to(device, dtype=dtype)
            if (lora_key + '.lora_down.weight') in lora_sd:
                # LoRA & LoCON
                assert (lora_key + '.lora_up.weight') in lora_sd

                down_weight = lora_sd.pop(lora_key + '.lora_down.weight').to(device, dtype=dtype)
                up_weight = lora_sd.pop(lora_key + '.lora_up.weight').to(device, dtype=dtype)
                dim = down_weight.size()[0]

                if len(weight.size()) == 2:
                    # linear
                    delta = up_weight @ down_weight
                else:
                    # conv2d
                    if (lora_key + '.lora_mid.weight') in lora_sd:
                        mid_weight = lora_sd.pop(lora_key + '.lora_mid.weight').to(device, dtype=dtype)
                        down_weight = merge_conv(mid_weight, down_weight.transpose(0,1)).transpose(0,1)
                        del mid_weight
                    delta = merge_conv(down_weight, up_weight)

                weight = (weight.to(device, dtype=dtype) + ratio * (alpha / dim * delta)).to('cpu', dtype=dtype)
                del down_weight, up_weight, delta
            elif (lora_key + '.hada_w1_a') in lora_sd:
                # LoHA
                assert (lora_key + '.hada_w1_b') in lora_sd
                assert (lora_key + '.hada_w2_a') in lora_sd
                assert (lora_key + '.hada_w2_b') in lora_sd

                w1a = lora_sd.pop(lora_key + '.hada_w1_a').to(device, dtype=dtype)
                w1b = lora_sd.pop(lora_key + '.hada_w1_b').to(device, dtype=dtype)
                w2a = lora_sd.pop(lora_key + '.hada_w2_a').to(device, dtype=dtype)
                w2b = lora_sd.pop(lora_key + '.hada_w2_b').to(device, dtype=dtype)
                dim = w1b.shape[0]

                if (lora_key + '.hada_t1') in lora_sd:
                    # conv2d 3x3
                    assert (lora_key + '.hada_t2') in lora_sd
                    t1 = lora_sd.pop(lora_key + '.hada_t1').to(device, dtype=dtype)
                    t2 = lora_sd.pop(lora_key + '.hada_t2').to(device, dtype=dtype)

                    delta = torch.einsum('i j k l, j r, i p -> p r k l',
                        t1, w1b, w1a
                    ) * torch.einsum('i j k l, j r, i p -> p r k l',
                        t2, w2b, w2a
                    )

                    del t1, t2
                else:
                    # conv2d 1x1 & linear
                    delta = (w1a @ w1b) * (w2a @ w2b)

                delta = delta.reshape(weight.shape)
                weight = (weight.to(device, dtype=dtype) + ratio * (alpha / dim * delta)).to('cpu', dtype=dtype)
                del w1a, w1b, w2a, w2b, delta
            base_sd[key] = weight
    return base_sd


def match_lora_dict(key, prefix):
    lora_key = None
    if key.endswith('.weight'):
        lora_key = key.replace('.weight', '')
        lora_key = lora_key.replace('.', '_')
        lora_key = prefix + lora_key
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
    # normalize to integer multiple of 64
    width, height = map(lambda x: max(x - (x % 64), 512), (width, height))
    return width, height


def resize_image(image, width, height):
    if image:
        w, h = image.size
        if w != width or h != height:
            factor = min(w / width, h / height)
            image = image.resize(
                (int(w / factor), int(h / factor)),
                resample=Image.Resampling.LANCZOS
            )
            w, h = image.size
            image = image.crop((
                (w - width) // 2,
                (h - height) // 2,
                (w + width) // 2,
                (h + height) // 2
            ))
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


def variant_mix(base_sd, variant_sd):
    for key, tensor in variant_sd.items():
        if key in base_sd:
            if key == 'conv_in.weight':
                tensor[:, 0:4, :, :] += base_sd[key]
            else:
                tensor += base_sd[key]
            base_sd[key] = tensor


class StableDiffusion:
    @torch.no_grad()
    def __call__(self, opt={}):
        start_time = time.time()

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
        self.load_model(opt.model, opt.vae, opt.lora, opt.model_variant, opt.control)
        self.set_schedular(opt.sampler)
        factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        channels = self.vae.config.latent_channels
        pix2pix_hack = hasattr(self.scheduler, 'sigmas') and opt.model_variant == 'instruct-pix2pix'
        
        model_loaded_time = time.time()

        # 3. Encode prompts
        with self.to_device(self.text_encoder):
            text_input = self.tokenizer(
                [opt.prompt] * opt.n_samples,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            )
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device),
                skip_last_n_layers=opt.clip_skip
            )
            if do_classifier_free_guidance:
                uncond_input = self.tokenizer(
                    [opt.neg_prompt] * opt.n_samples,
                    padding='max_length',
                    max_length=text_input.input_ids.shape[-1],
                    truncation=True,
                    return_tensors="pt"
                )
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids.to(self.device),
                    skip_last_n_layers=opt.clip_skip
                )
                text_embeddings = torch.cat(
                    [text_embeddings, uncond_embeddings, uncond_embeddings]
                ) if opt.model_variant == 'instruct-pix2pix' else torch.cat(
                    [uncond_embeddings, text_embeddings]
                )

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

        # 6. Prepare init image and mask
        init_mask = None
        latent_mask = None
        if opt.mask is not None:
            init_mask = resize_image(opt.file[opt.mask], width, height)
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

                init_image = resize_image(opt.file[opt.image], width, height)
                init_image = create_np_image(init_image)
                init_image = init_image.to(self.device, dtype=self.dtype)

                init_latent = self.vae.encode(init_image).latent_dist.sample()
                init_latent = init_latent * self.vae.config.scaling_factor

                if opt.boost != 0.0:
                    print('>> Use latent image upscaler')
                    tmp_image = crop_image(opt.file[opt.image], width, height)
                    if tmp_image.size >= (width, height):
                        tmp_image = tmp_image.resize(
                            (width // 2, height // 2),
                            resample=Image.Resampling.LANCZOS
                        )
                    tmp_image = create_np_image(tmp_image)
                    tmp_image = tmp_image.to(self.device, self.dtype)

                    tmp_latent = self.vae.encode(tmp_image).latent_dist.sample()
                    tmp_latent = torch.nn.functional.interpolate(
                        tmp_latent,
                        size=(height // factor, width // factor),
                        mode='bilinear',
                        antialias=False
                    )
                    tmp_latent = tmp_latent * self.vae.config.scaling_factor
                    tmp_latent = init_latent * (1. - opt.boost) + tmp_latent * opt.boost

                    if latent_mask is not None:
                        init_latent = init_latent * latent_mask + (1. - latent_mask) * tmp_latent
                    else:
                        init_latent = tmp_latent

                if opt.model_variant == 'inpainting' and init_mask is not None:
                    masked_image = init_image * (init_mask > 0.5)
                    masked_latent = self.vae.encode(masked_image).latent_dist.sample()
                    masked_latent = masked_latent * self.vae.config.scaling_factor
                    image_condition = torch.cat([
                        # Inpainting use white mask instead of transparent
                        1 - latent_mask[:, 0:1, :, :],
                        masked_latent
                    ], dim=1)
                elif opt.model_variant == 'instruct-pix2pix':
                    image_condition = self.vae.encode(init_image).latent_dist.mode()

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

        if image_condition is not None:
            image_condition = image_condition.repeat(opt.n_samples, 1, 1, 1)
            image_condition = torch.cat(
                [
                    image_condition,
                    image_condition,
                    torch.zeros_like(image_condition)
                ] if opt.model_variant == 'instruct-pix2pix' else [
                    image_condition,
                    image_condition
                ]
            ) if do_classifier_free_guidance else image_condition

        # 7. Generate controlnet condition image
        control = []
        for i in range(len(opt.control)):
            controlnet_cond = resize_image(opt.file[opt.control[i][0]], width, height)
            controlnet_cond_scale = opt.control[i][3]
            controlnet_cond = self.annotator.annotate(
                opt.control[i][1],
                controlnet_cond
            )
            # print(
            #     f'>> DEBUG: writing the control condition to cond{i}.png'
            # )
            # control_condition.save(f'cond{i}.png')
            controlnet_cond = create_np_image(controlnet_cond, normalize=False)
            if opt.control[i][2] == 'inpaint' and init_mask is not None:
                controlnet_cond[init_mask < 0.5] = -1.0
            controlnet_cond = controlnet_cond.to(self.device, dtype=self.dtype)
            controlnet_cond = controlnet_cond.repeat(opt.n_samples, 1, 1, 1)
            controlnet_cond = torch.cat(
                [controlnet_cond] * (3 if opt.model_variant == 'instruct-pix2pix' else 2)
            ) if do_classifier_free_guidance else controlnet_cond
            control.append({
                'cond': controlnet_cond,
                'scale': controlnet_cond_scale
            })

        # 8. Denoising loop
        # Use tiled UNet to process large latent
        # Because it's relatively hard to ensure the consistance between tiles,
        # try to increase max tile size to reduce the tile numbers
        full_latent = latent
        full_noise_pred = torch.zeros_like(latent, device=self.device, dtype=self.dtype)
        full_noise_pred = torch.cat(
            [ full_noise_pred ] * (3 if opt.model_variant == 'instruct-pix2pix' else 2)
        ) if do_classifier_free_guidance else full_noise_pred
        tiles = create_tiles(
            full_latent,
            full_noise_pred,
            max_size=(2048 if opt.tile is None else opt.tile[0]) // factor,
            min_size=512 // factor,
            overlap=(64 if opt.tile is None else opt.tile[1]) // factor,
        )
        n_y = len(tiles)
        n_x = len(tiles[0])
        if n_y * n_x > 1:
            print(f'>> Use tiled UNet with {n_y}x{n_x} tiles')

        # Start!
        num_warmup_steps = len(timesteps) - steps * self.scheduler.order
        with self.to_device(self.unet, *self.controlnet[:len(opt.control)]), tqdm(total=steps) as pbar:
            for i, t in enumerate(timesteps):
                for m in range(n_y):
                    for n in range(n_x):
                        # Get current tile and corresponding scheduler
                        tile = tiles[m][n]
                        latent = tile['latent']
                        y = tile['position'][0]
                        x = tile['position'][1]
                        h = tile['size'][0]
                        w = tile['size'][1]

                        # expand the latents if we are doing classifier free guidance
                        # The latents are expanded 3 times because for pix2pix the guidance\
                        # is applied for both the text and the input image.
                        latent_model_input = torch.cat(
                            [latent] * (3 if opt.model_variant == 'instruct-pix2pix' else 2)
                        ) if do_classifier_free_guidance else latent

                        # concat latents, image_condition in the channel dimension
                        scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        if image_condition is not None:
                            scaled_latent_model_input = torch.cat(
                                [scaled_latent_model_input, image_condition[:,:,y:y+h,x:x+w]],
                                dim=1
                            )

                        # use controlnet
                        down_block_res_samples, mid_block_res_sample = (None, None)
                        for j in range(len(control)):
                            down_samples, mid_sample = self.controlnet[j](
                                scaled_latent_model_input,
                                t,
                                encoder_hidden_states=text_embeddings,
                                controlnet_cond=control[j]['cond'][:,:,y*factor:(y+h)*factor,x*factor:(x+w)*factor],
                                conditioning_scale=control[j]['scale'],
                                guess_mode=not do_classifier_free_guidance,
                                return_dict=False
                            )
                            if j == 0:
                                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
                            else:
                                down_block_res_samples = [
                                    samples_prev + samples_curr
                                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                                ]
                                mid_block_res_sample += mid_sample

                        # predict the noise residual
                        noise_pred = self.unet(
                            scaled_latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
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

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if pix2pix_hack:
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if do_classifier_free_guidance:
                    if opt.model_variant == 'instruct-pix2pix':
                        # proper range of image guidance scale is [2.0 to 1.5]
                        image_guidance_scale = 2 - abs(opt.strength) / 2
                        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + opt.scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + opt.scale * (noise_pred_text - noise_pred_uncond)

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if pix2pix_hack:
                    noise_pred = (noise_pred - latent) / (-sigma)

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
                if latent_mask is not None and opt.model_variant != 'inpainting':
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

        # 9. Decode latent
        with self.to_device(self.vae):
            latent = (1 / self.vae.config.scaling_factor) * latent
            x_samples = self.vae.decode(latent).sample
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
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

        print(f'>> Model loading costs {model_loaded_time - start_time:.2f}s, inference costs {inference_complete_time - model_loaded_time:.2f}s, postprocess costs {time.time() - inference_complete_time:.2f}s')

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
            'clip_skip': opt.clip_skip,
            'model': opt.model,
            'vae': opt.vae,
            'lora': opt.lora,
            'control': opt.control
        }

    @torch.no_grad()
    def __init__(self, base, index, annotator, max_controlnets, device, dtype=torch.float16):
        self.base = base
        self.dict = index
        self.annotator = annotator
        self.device = device
        self.dtype = dtype

        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            self.base['root'],
            subfolder='feature_extractor',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            self.base['root'],
            subfolder='safety_checker_x',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.scheduler = diffusers.PNDMScheduler.from_pretrained(
            self.base['root'],
            subfolder='scheduler',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.unet = UNetWrapper.from_pretrained(
            self.base['root'],
            subfolder='unet',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.text_encoder = TextEncoderWrapper.from_pretrained(
            self.base['root'],
            subfolder='text_encoder',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.tokenizer = TokenizerWrapper.from_pretrained(
            self.base['root'],
            subfolder='tokenizer',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.vae = AutoencoderKL.from_pretrained(
            self.base['root'],
            subfolder='vae',
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        self.controlnet = [ diffusers.ControlNetModel.from_pretrained(
            os.path.join(self.base['root'], 'controlnet'),
            torch_dtype=self.dtype,
            use_safetensors=True
        ) ]

        for i in range(1, max_controlnets):
            self.controlnet.append(
                copy.deepcopy(self.controlnet[0])
            )

        self._last_loaded_ckpt = None
        self._last_loaded_vae = None
        self._last_mixed_lora = None
        self._last_used_variant = None
        self._last_used_control = None

        self.load_embeddings()
        self.apply_optimization()

        self.default_args = {
            'model': None,
            'model_variant': None,
            'vae': None,
            'lora': [],
            'control': [],
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
            'clip_skip': 1,
            'tile': None
        }

    def apply_optimization(self):
        if str(self.device) == 'cuda':
            print('>> Enbalbe xformers')
            self.unet.enable_xformers_memory_efficient_attention()
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

    def exist(self, model=None, vae=None, lora=None, variant=None, control=None):
        if model and not model in self.dict['ckpt']:
            return False
        if vae and not vae in self.dict['vae']:
            return False
        if lora and not lora in self.dict['lora']:
            return False
        if variant and not variant in self.base['variant']:
            return False
        if control and not control in self.base['controlnet']:
            return False
        return True

    def get_controlnet_names(self):
        return list(self.base['controlnet'].keys())

    def get_embedding_names(self):
        return list(self.dict['emb'].keys())

    def get_lora_names(self):
        return list(self.dict['lora'].keys())

    def get_model_names(self):
        return list(self.dict['ckpt'].keys())

    def get_vae_names(self):
        return list(self.dict['vae'].keys())

    def get_schedular_names(self):
        return list(SAMPLER_CHOICES)

    def load_embeddings(self):
        embedding_db = {}
        for name, pathname in self.dict['emb'].items():
            data = torch.load(pathname, map_location='cpu')
            # stable diffusion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")
            embedding_db[name] = emb.detach()

        self.text_encoder.register_embeddings(
            self.tokenizer.register_tokens(embedding_db)
        )

    def load_model(self, ckpt, vae, lora=[], variant=None, control=[]):
        ckpt = self.dict['ckpt'][ckpt]
        vae = self.dict['vae'][vae]
        lora = list(map(lambda x: (self.dict['lora'][x[0]], x[1]), lora))

        unet_reloaded = False
        if self._last_loaded_ckpt != ckpt or self._last_mixed_lora != lora or self._last_used_variant != variant:
            unet_sd = load_state_dict(ckpt['unet'])
            text_encoder_sd = load_state_dict(ckpt['text_encoder'])

            for pathname, ratio in lora:
                lora_sd = load_state_dict(pathname)
                lora_mix(unet_sd, lora_sd, 'lora_unet_', ratio, self.device, self.dtype)
                lora_mix(text_encoder_sd, lora_sd, 'lora_te_', ratio, self.device, self.dtype)
                if len(lora_sd) > 0:
                    print(">> Unexpected LoRA keys:")
                    print(lora_sd.keys())

            if variant is not None:
                print(f'>> Use variant model: {variant}')
                variant_unet_sd = load_state_dict(self.base['variant'][variant])
                variant_mix(unet_sd, variant_unet_sd)

            self.unet.load_state_dict(unet_sd, strict=False)
            self.text_encoder.load_state_dict(text_encoder_sd, strict=False)
            unet_reloaded = True

        control = [ v[2] for v in control ]
        if self._last_used_control != control or unet_reloaded:
            for i in range(len(control)):
                control_name = control[i]
                controlnet_sd = load_state_dict(self.base['controlnet'][control_name])
                controlnet_mix(controlnet_sd, self.unet.state_dict())
                self.controlnet[i].conv_in = copy.deepcopy(self.unet.conv_in)
                self.controlnet[i].load_state_dict(controlnet_sd)

        if self._last_loaded_vae != vae:
            vae_sd = load_state_dict(vae)
            self.vae.load_state_dict(vae_sd, strict=False)

        self._last_loaded_ckpt = ckpt
        self._last_loaded_vae = vae
        self._last_mixed_lora = lora
        self._last_used_variant = variant
        self._last_used_control = control

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


class TextEncoderWrapper(CLIPTextModel):
    def __init__(self, config):
        super().__init__(config)
        self._extra_embeddings = None

    def forward(self, tokens, skip_last_n_layers=1, **kwargs):
        if skip_last_n_layers == 1:
            outputs = super().forward(
                input_ids=tokens,
                **kwargs
            )
            z = outputs.last_hidden_state
        else:
            if 'output_hidden_states' in kwargs:
                del kwargs['output_hidden_states']
            if 'return_dict' in kwargs:
                del kwargs['return_dict']
            outputs = super().forward(
                input_ids=tokens,
                output_hidden_states=True,
                return_dict = True,
                **kwargs
            )
            z = outputs.hidden_states[-skip_last_n_layers]
            z = self.text_model.final_layer_norm(z)
        return z

    def load_state_dict(self, sd, **kwargs):
        if 'text_model.embeddings.token_embedding.weight' in sd:
            if self._extra_embeddings is not None:
                weight = sd['text_model.embeddings.token_embedding.weight']
                weight = torch.cat([ weight, self._extra_embeddings ])
                sd['text_model.embeddings.token_embedding.weight'] = weight
        return super().load_state_dict(sd, **kwargs)

    def register_embeddings(self, extra_embeddings):
        i = self.get_input_embeddings().weight.shape[0]
        n = len(extra_embeddings)
        self.resize_token_embeddings(i + n)

        embeddings = self.get_input_embeddings()
        for embedding in extra_embeddings:
            embeddings.weight.data[i] = embedding
            i += 1

        self._extra_embeddings = torch.stack(extra_embeddings)


class TokenizerWrapper(CLIPTokenizer):
    def __call__(self, text, **kwargs):
        for i in range(len(text)):
            text[i] = re.sub(r'<(.*?)>', lambda x: '<' + self._unfold(x.group(1)) + '>', text[i])
        return super().__call__(text, **kwargs)

    def _unfold(self, name):
        if name in self.embedding_db.keys():
            length = self.embedding_db[name].shape[0]
            return '><'.join([ f'{name}#{i}' for i in range(length) ])
        else:
            return name

    def register_tokens(self, embedding_db):
        self.embedding_db = embedding_db

        vocab = self.get_vocab()
        tokens = []
        embeddings = []

        for name, embedding in self.embedding_db.items():
            for i in range(embedding.shape[0]):
                token = f'<{name}#{i}>'
                assert not token in vocab, f'Token {token} already in tokenizer vocabulary.'
                tokens.append(token)
                embeddings.append(embedding[i])

        self.add_tokens(tokens)

        return embeddings


class UNetWrapper(UNet2DConditionModel):
    def load_state_dict(self, sd, **kwargs):
        in_channels = sd['conv_in.weight'].shape[1]
        block_out_channels = self.config['block_out_channels']
        conv_in_kernel = self.config['conv_in_kernel']
        conv_in_padding = (conv_in_kernel - 1) // 2
        dtype = self.conv_in.weight.dtype

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        ).to(dtype)

        return super().load_state_dict(sd, **kwargs)
