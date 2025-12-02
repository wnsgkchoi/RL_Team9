import torch

class BaseAgent:
    """
    This is a base class that contains the methods every agent should implement
    """
    action_generation_cache = {}
    def get_lora_module_names():
        raise NotImplementedError("Subclasses must implement get_lora_module_names method.")
    
    def get_value(self, x, value_prompt_template='{}', text_description=None, **kwargs):
        raise NotImplementedError("Subclasses must implement get_value method.")

    def get_action_and_value(self, x, value_prompt_template='{}', action_template='{}', action=None, text_description=None, **kwargs):
        raise NotImplementedError("Subclasses must implement get_action_and_value method.")

    @torch.no_grad()
    def predict(self, x):
        output = self.get_action_and_value(x, action=None, text_description=[None]*x.shape[0])
        action = output['action_logits'].argmax(dim=-1)
        return action
