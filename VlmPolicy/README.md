# VlmPolicyEsann24
This is the code repository for the paper titled "Vision Language Models as Policy Learners in Reinforcement Learning Environments" presented at Esann 2024.

Create a conda env and downgrade setuptools and wheel for later gym 0.21.0 compatibility:
```bash
conda create -n crafter_env python=3.10
conda activate crafter_env

pip install setuptools==65.5.0 wheel==0.38.4
```

Then clone the SmartPlay repo for environment support:
```bash
git clone https://github.com/giobin/SmartPlay.git
cd SmartPlay
pip install -e .
```

How to install:
```bash
[git clone https://github.com/giobin/VlmPolicyEsann24.git
cd VlmPolicyEsann24
pip install -r requirements.txt
```
