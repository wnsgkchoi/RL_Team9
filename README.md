# RL Team 9 Project  

This workspace contains the implementation for the team project of Team 9.  

## 📂 Workspace Structure  

The workspace is organized into two main repositories:

```text
CSED627/
├── VlmPolicy/                  # Main project code (Agents, Training, Inference)
│   ├── configs/                # Hydra configuration files
│   ├── src/                    # Source code
│   └── ...
└── SmartPlay/                  # Environment dependency (Crafter, etc.)
    ├── src/
    └── ...
```

## 🛠️ Installation & Setup

To run the code in `VlmPolicy`, you need to set up a specific Conda environment and install dependencies with careful version control.

### 0. Clone Repository  

```bash
git clone <URL>
```

### 1. Create Conda Environment

```bash
conda create -n crafter_env python=3.10
conda activate crafter_env
```

### 2. Install Dependencies

**Step 1: Downgrade build tools for compatibility**
```bash
pip install setuptools==65.5.0 wheel==0.38.4 pip==24.0.0
```

**Step 2: Install SmartPlay (Environment Support)**
```bash
cd SmartPlay
pip install -e .
cd ..
```

**Step 3: Install Project Requirements & Fix Versions**
Navigate to the main project folder:
```bash

pip install -r requirements.txt
```

## 🚀 How to Run

All scripts should be run from the `VlmPolicy/src` directory.

### 1. Training (PPO)
To train an agent (e.g., IDEFICS) using PPO:

```bash
# To assign proper GPU in manual
simple_gpu_scheduler --gpus [gpu] < train_ppo.txt
```

### 2. Inference / Evaluation
To evaluate a trained model or run inference:

```bash
cd VlmPolicy/src
python inference.py --config-name idefics_esann_base_config
```

### 3. Baseline (DQN)
To run the DQN baseline on Crafter:

```bash
cd VlmPolicy/src
python dqn.py --env-id CrafterReward-v1
```

## ⚙️ Configuration

The project uses **Hydra** for configuration. You can find and modify config files in `VlmPolicy/configs/`.

- `idefics_esann_base_config.yaml`: Default IDEFICS agent config.
- `cnn_esann_base_config.yaml`: CNN baseline config.
- `vlm_esann_base_config.yaml`: General VLM config.

You can override parameters via command line, e.g.:
```bash
python ppo_crafter_parallel.py --config-name idefics_esann_base_config total_timesteps=100000
```

## 📖 References  

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  

