# Tmp  

## Per one Update

|env|process|size|run time|
|-|-|-|-|
|naive|direct|8|21:24 * 100|
|gpt|direct|8|15:50 * 100|
|Manual|direct|8|20:43 * 100|
|naive|curriculum|4|22:18 * 100|
|naive|curriculum|6|??:?? * 100| 
|naive|curriculum|8|23:16 * 100|
|manual|curriculum|4|23:14 * 100|
|manual|curriculum|6| * 100|
|manual|curriculum|8|21:55 * 100|
|gpt|curriculum|4| * 100|
|gpt|curriculum|6| * 100|
|gpt|curriculum|8| * 100|



### Appendix: Commands

```
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py --config-name idefics_esann_base_config  gpu_device=7 env_area=8 total_timesteps=2048 wandb_project_name="frozenlake_tmp" wandb_log_dir="naive_8x8_vanilla"
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py --config-name idefics_esann_base_config gpu_device=7 env_id=FrozenLakeText-Potential-v0 env_area=8 total_timesteps=2048 wandb_project_name="frozenlake_tmp" wandb_log_dir="naive_8x8_potential"
CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py  --config-name idefics_esann_base_config  gpu_device=7 env_area=4 total_timesteps=2048 wandb_project_name="frozenlake_tmp" wandb_log_dir="stage1_4x4"
CUDA_VISIBLE_DEVICES=5 python VlmPolicy/src/ppo_crafter_parallel_v2.py --config-name idefics_esann_base_config gpu_device=5 env_area=6 model_ckpt="runs/FrozenLakeText-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4__20251209_123135/agent_ckpt_last_gs-40960/" total_timesteps=2048 wandb_project_name="frozenlake_tmp" wandb_log_dir="stage2_6x6_vanilla"

CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_area=8 \
    model_ckpt="checkpoint/vanilla_6*6/" \
    total_timesteps=2048 \
    wandb_project_name="frozenlake_tmp" \
    wandb_log_dir="stage3_8x8_vanilla" 

CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=4 \
    total_timesteps=2048 \
    wandb_project_name="frozenlake_tmp" \
    wandb_log_dir="stage1_4x4_potential"

CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=6 \
    model_ckpt="runs/FrozenLakeText-Potential-v0__HuggingFaceM4-idefics-9b-instruct__ppo_crafter_parallel_v2__9812__stage1_4x4_potential__20251209_081755/agent_ckpt_last_gs-40960/" \
    total_timesteps=2048 \
    wandb_project_name="frozenlake_tmp" \
    wandb_log_dir="stage2_6x6_potential"

CUDA_VISIBLE_DEVICES=7 python VlmPolicy/src/ppo_crafter_parallel_v2.py \
    --config-name idefics_esann_base_config \
    gpu_device=7 \
    env_id=FrozenLakeText-Potential-v0 \
    env_area=8 \
    model_ckpt="checkpoint/potential_6*6/" \
    total_timesteps=2048 \
    wandb_project_name="frozenlake_tmp" \
    wandb_log_dir="stage3_8x8_potential"
```

## 할 것  

1. 코드 수정 (GPT reward map 받아오기)  
2. naive, curriculum 실험 돌리기  
