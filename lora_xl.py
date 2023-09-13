# generate.py
from diffusers import StableDiffusionXLPipeline
import torch

xl_model_cfg = {
        'v1': '/home/dell/workspace/models/stable-diffusion-v1-5', # Use safety checker from v1
        'base': '/home/dell/workspace/models/stable-diffusion-xl-base-1.0',
        'refiner': '/home/dell/workspace/models/stable-diffusion-xl-refiner-1.0'
    }

# model_path = "sdxl-lora/checkpoint-1000"  # 根据自己设置的训练策略找到保存权重的checkpoint文件夹
model_path = "/home/dell/workspace/models/pixel-art-xl.safetensors"
pipe = pipe = StableDiffusionXLPipeline.from_pretrained(xl_model_cfg["base"], torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe.to("cuda")
pipe.load_lora_weights(model_path)

prompt = "A moose in watercolor painting style"  # 生成内容
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
# image.save("moose.png")  # 保存图像名
image.resize(size=(512, 512))




# ---------------------sd 15 lora --------------------------
#!/usr/bin/env python3
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

#   Diffusers提供了load_attn_procs()方法，可以将LoRA权重加载到模型的注意力层中
UNet2DConditionModel.load_attn_procs

# unet.save_attn_procs()  将注意力层参数保存起来


pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

pipeline.unet.load_attn_procs("patrickvonplaten/lora")
pipeline.to("cuda")

prompt = "A photo of sks dog in a bucket"

images = pipeline(prompt, num_images_per_prompt=4).images
    
for i, image in enumerate(images):
    image.save(f"/home/patrick_huggingface_co/images/dog_{i}.png")


