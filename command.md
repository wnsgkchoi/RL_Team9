# Commands  

## Direct  

### Vanilla  

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py  --config-name idefics_esann_base_config  gpu_device=7 env_area=8 total_timesteps=163840 wandb_project_name="frozenlake_curriculum" wandb_log_dir="naive_8x8_vanilla"
```

### potential

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=6 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=8 \
    total_timesteps=163840 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="naive_8x8_potential"
```






## Curriculum Learning  

### Vanilla  

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py  --config-name idefics_esann_base_config  gpu_device=6 env_area=4 total_timesteps=81920 wandb_project_name="frozenlake_curriculum" wandb_log_dir="stage1_4x4"
```

```bash
CUDA_VISIBLE_DEVICES=5 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=5 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4__20251208_011248/agent_ckpt_last_gs-40960/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_vanilla"
```

### penalty  

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py --config-name idefics_esann_base_config gpu_device=6 env_id=FrozenLakeText-Penalty-v0 env_area=4 total_timesteps=81920 wandb_project_name="frozenlake_curriculum" wandb_log_dir="stage1_4x4_penalty"
```

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=6 \
    env_id=FrozenLakeText-Penalty-v0 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-Penalty-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4_penalty__20251208_011249/agent_ckpt_last_gs-40960/" \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_penalty"
```

### Potential  

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=4 \
    total_timesteps=81920 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage1_4x4_potential"
```