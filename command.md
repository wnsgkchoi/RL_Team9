# Commands  

## Direct  

### Vanilla  

```bash
CUDA_VISIBLE_DEVICES=3 python VlmPolicy/src/ppo_crafter_parallel_v2.py  --config-name idefics_esann_base_config  gpu_device=3 env_area=8 total_timesteps=204800 wandb_project_name="frozenlake_curriculum" wandb_log_dir="naive_8x8_vanilla"
```

### potential

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=8 \
    total_timesteps=204800 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="naive_8x8_potential"
```

### gpt  

```bash
CUDA_VISIBLE_DEVICES=3 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=3 \
    env_id=FrozenLakeText-GPT-v0 \
    env_area=8 \
    total_timesteps=204800 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="naive_8x8_gpt_reward"
```


## Curriculum Learning  

### Vanilla  

#### 4*4

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py  --config-name idefics_esann_base_config  gpu_device=7 env_area=4 total_timesteps=40960 wandb_project_name="frozenlake_curriculum" wandb_log_dir="stage1_4x4"
```

#### 6*6  

```bash
CUDA_VISIBLE_DEVICES=5 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=5 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4__20251209_123135/agent_ckpt_last_gs-40960/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_vanilla"
```

#### 8*8

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=6 \
    env_area=8 \
    model_ckpt="checkpoint/vanilla_6*6/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage3_8x8_vanilla"
```



### Potential  

#### 4*4  

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=4 \
    total_timesteps=40960 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage1_4x4_potential"
```

#### 6*6  

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-Potential-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4_potential__20251209_081755/agent_ckpt_last_gs-40960/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_potential"
```

#### 8*8

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=8 \
    model_ckpt="checkpoint/potential_6*6/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage3_8x8_potential"
```

### GPT  

```bash
CUDA_VISIBLE_DEVICES=2 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=2 \
    env_id=FrozenLakeText-GPT-v0 \
    env_area=4 \
    total_timesteps=40960 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage1_4x4_gpt_reward"
```

```bash
CUDA_VISIBLE_DEVICES=2 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=2 \
    env_id=FrozenLakeText-GPT-v0 \
    env_area=6 \
    model_ckpt="checkpoint/gpt_4*4/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_gpt_reward"
```

CUDA_VISIBLE_DEVICES=2 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=2 \
    env_id=FrozenLakeText-GPT-v0 \
    env_area=8 \
    model_ckpt="checkpoint/gpt_6*6/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage3_8x8_gpt_reward"
```

```bash
CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port=29500 \
    VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    env_id=FrozenLakeText-GPT-v0 \
    env_area=6 \
    model_ckpt="checkpoint/gpt_4*4/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_gpt_reward"
```