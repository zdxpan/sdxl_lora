# SDXL  kohya sdXL 训练

需要下载 kohya_ss  修改目录中 sdxl_train_network 进行训练~

准备SDXL 主模型依赖
'/home/dell/workspace/models/stable-diffusion-xl-base-1.0'


# 数据构造

lora准备数据
../kohya_ss/jl_test
├── jl_test_face.png
├── jl_test.jpg_bac
└── jl_test_pure.png

模型训练输出~
../kohya_ss/output
├── jl_out.safetensors
├── jl_out-step00000050.safetensors
└── jl_out-step00000100.safetensors

jl.toml 数据构造脚本~
`[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# これは DreamBooth 方式のデータセット
[[datasets]]
resolution = [768, 768]
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '/home/dell/workspace/kohya_ss/jl_test'
  class_tokens = 'jl'
  # このサブセットは keep_tokens = 2 （所属する datasets の値が使われる）
`

SDXL训练脚本修改：
python   sdxl_train_network.py   该脚本尽量不用 GUi启动，依赖较多，环境配置麻烦~

`    args.pretrained_model_name_or_path = '/home/dell/workspace/models/stable-diffusion-xl-base-1.0'
    # args. "/home/dell/workspace/models/stable-diffusion-xl-base-1.0/vae_fp16_fix",
    args.dataset_config = "jl.toml"
    args.output_dir = "output"
    args.output_name = "jl_out"
    args.save_model_as = "safetensors"
    args.prior_loss_weight = 1.0
    args.max_train_steps = 100
    args.learning_rate = 1e-3
    args.optimizer_type="AdamW8bit" 
    args.xformers=False
    args.mixed_precision="fp16" 
    args.cache_latents =True
    args.resolution = "768,768"
    args.gradient_checkpointing=True
    # args.save_every_n_epochs=1
    args.save_every_n_steps = 50
    args.network_module="networks.lora"
    args.network_dim = 64
    args.network_alpha = 16
    args.vae="/home/dell/workspace/models/stable-diffusion-xl-base-1.0/"
`
