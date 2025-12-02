import torch
import torch.nn as nn
import numpy as np
import os, sys

from math import floor
from torch.distributions.categorical import Categorical

from accelerate import Accelerator

accelerator = Accelerator()

rocket_dir = os.path.split(os.path.abspath(os.curdir))[:-1]
rocket_src_dir = os.path.join(*rocket_dir, 'src')
rocket_dir = os.path.join(*rocket_dir)

print(f'Appending to sys.path:', [rocket_dir, rocket_src_dir])

sys.path.append(rocket_dir)
sys.path.append(rocket_src_dir)

from src.environments.frozen_env import make_frozen_env, FrozenActions
from src.agents.Mistral_agent import MistralAgent

agent = MistralAgent("mistralai/Mistral-7B-Instruct-v0.2",
                        action_enum=FrozenActions,
                         is_lora=True, 
                         padding_side="right", 
                         gradient_ckpt=False,
                         disable_adapters_for_generation=True)
model = accelerator.prepare(agent)
print(f"{accelerator.device=}, {accelerator.num_processes=}, {accelerator.process_index=}, {accelerator.is_local_main_process=}, {accelerator.is_main_process=}\n")

in_descriptions = ["this is a given prompt", "this is another one"]
res = model.module.get_value(x=None, text_description=in_descriptions)
print(f"{accelerator.device=}, {accelerator.num_processes=}, {accelerator.process_index=}, {accelerator.is_local_main_process=}, {accelerator.is_main_process=}\n{res=}")