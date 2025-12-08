# Commands  

## Curriculum Learning  

### no penalty  

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_area=4 \
    total_timesteps=40960 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage1_4x4"
```

### penalty  

```bash
CUDA_VISIBLE_DEVICES=6 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=6 \
    env_id=FrozenLakeText-Penalty-v0 \
    env_area=4 \
    total_timesteps=40960 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage1_4x4_penalty"
```

```bash
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Penalty-v0 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-Penalty-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4_penalty/agent_ckpt_last_gs-10240/" \
    total_timesteps=20480 \
    wandb_project_name="frozenlake_curriculum" \
    wandb_log_dir="stage2_6x6_penalty"
```