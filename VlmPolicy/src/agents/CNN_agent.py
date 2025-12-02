import torch.nn as nn
import numpy as np
import torch
from math import floor
from torch.distributions.categorical import Categorical
from agents.base_agent import BaseAgent


def compute_final_layer_dimension(in_dim):
  return floor(floor((floor((in_dim - 8)/4 + 1) - 4) / 2 + 1) -2)

class CNNAgent(nn.Module, BaseAgent):
    def __init__(self, observation_space, action_number):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
        )
        with torch.no_grad():
            self._process_input(torch.zeros(1, *observation_space.shape))
            
        self.actor = nn.Linear(512, action_number)
        self.critic = nn.Linear(512, 1)

        self._initialize_weights()  # Initialize weights

    def get_lora_module_names(self):
        return 'No LORA'
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def get_value(self, x, text_description=None, **kwargs):
        output = {'actions':None, 'log_prob':None, 'entropy':None, 'values':self.critic(self._process_input(x))}
        return output

    def get_action_and_value(self, x, action=None, text_description=None, temperature=1.0, **kwargs):
        hidden = self._process_input(x)
        logits = self.actor(hidden)

        # apply temperature
        if temperature == 'max_logit':
            logits = logits - logits.max(-1, keepdim=True).values
        elif isinstance(temperature, float):
            logits = logits / temperature
        else:
            raise Exception(f"Invalid temperature: {temperature}")

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        output = {'action':action, 'log_prob':probs.log_prob(action), 'entropy':probs.entropy(), 'values':self.critic(hidden), 'action_logits':logits}
        return output
    
    def _process_input(self, x):
        # input is [num_env, 64,64,3], and must become [num_env, 3,64,64]
        x = x.permute(0,3,1,2)
        x = x.float() / 255.0  # Normalize input
        return self.network(x)