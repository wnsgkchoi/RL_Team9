import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from torch.distributions.categorical import Categorical
from accelerate import Accelerator

from agents.base_agent import BaseAgent

accelerator = Accelerator()

class PHI2Agent(nn.Module, BaseAgent):
    def __init__(
        self, checkpoint_name, action_enum, is_lora=False, padding_side="right", gradient_ckpt=False
    ):
        super().__init__()
        self.action_enum = action_enum
        self.network, self.tokenizer, self.config = self._load_model(
            checkpoint_name, is_lora, padding_side, gradient_ckpt
        )
        # the critic remains in float32. (i.e. no lora or quantized stuff)
        self.critic = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1024),  # Input layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(1024, 512),  # Hidden layer 1
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, 1),  # Output layer
        )

    def get_lora_module_names(self):
        names = self.network.targeted_module_names
        return "\n".join(names)

    def get_choices_logits(self, logits, labels, attention_mask, context_prompt_lens):

        # select only actions
        logits = logits[..., context_prompt_lens - 1 :, :]
        attention_mask = attention_mask[..., context_prompt_lens - 1 :]
        labels = labels[..., context_prompt_lens - 1 :]

        shift_logits = logits[..., :-1, :]
        shift_attention = attention_mask[..., 1:]
        shift_labels = labels[..., 1:]

        tokens_logprobs = torch.gather(
            shift_logits, 2, shift_labels[:, :, None]
        ).squeeze(-1)
        tokens_logprobs_applied_mask = tokens_logprobs * shift_attention

        # Calculate the sum and count of unmasked elements
        sum_unmasked = tokens_logprobs_applied_mask.sum(dim=1)
        count_unmasked = shift_attention.sum(dim=1)

        # Calculate the mean
        mean_tokens_logprobs_unmasked = sum_unmasked / count_unmasked
        return mean_tokens_logprobs_unmasked

    def create_prompt_for_action_and_value(self, just_value=False, text_description=None):
        # text_description is None for the first step (after reset()) and if use_text_description is set to False.

        v_prompts = [f"Instruct: You are an agent in a survival 2D game. {t} What's the next best action?\nOutput:" if t is not None else "Instruct: You are an agent in a survival 2D game. What's the next best action?\nOutput:" for t in text_description]

        prompts = []
        for vp in v_prompts:
            if not just_value:
                for action in self.action_enum:
                    v = vp + f" {action.value}"
                    prompts.append(v)
            else:
                prompts = v_prompts
        return prompts, v_prompts

    def get_value(self, x, text_description=None):
        return self.get_action_and_value_one_forward(action=None, just_value=True, text_description=text_description)

    def get_action_and_value(self, x, action=None, text_description=None, temperature=1.0):
        return self.get_action_and_value_one_forward(action=action, just_value=False, text_description=text_description, temperature=temperature)

    def get_action_and_value_one_forward(self, action=None, just_value=False, text_description=None, temperature=1.0):
        output = {"actions": None, "log_prob": None, "entropy": None, "values": None}

        # step in common for actions and values computation
        batch_dim = len(text_description)

        prompts, just_context_prompt = self.create_prompt_for_action_and_value(just_value=just_value, text_description=text_description)

        context_prompt = self.tokenizer(just_context_prompt, return_tensors="pt", padding=True)
        input_processed = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.network.device)

        # get the context len, useful to locate the action tokens when computing the logits
        context_prompt_lens = torch.sum(context_prompt["attention_mask"], dim=1)

        padding_on_left_c = (context_prompt["attention_mask"][..., 0] == 0).sum() > (context_prompt["attention_mask"][..., -1] == 0).sum()
        padding_on_left_j = (input_processed["attention_mask"][..., 0] == 0).sum() > (input_processed["attention_mask"][..., -1] == 0).sum()

        assert not padding_on_left_c
        assert not padding_on_left_j

        result = self.network(**input_processed, output_hidden_states=True)

        if just_value:
            cls_token_hidden = result.hidden_states[-1][:, - 1]
        else:
            cls_token_hidden_deduplicated = result.hidden_states[-1][::len(self.action_enum)] # select the 0th, 17th, 34th, ... rows
            cls_token_hidden = cls_token_hidden_deduplicated[torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]

        values = self.critic(cls_token_hidden.float())
        output["values"] = values

        if just_value:
            return output

        # if more than 1 local_env we need to get_choices_logits for every one of them
        chunk_logits = result.logits.tensor_split(batch_dim)                              #[34, 39, 32002] -> #([17, 39, 32002], [17, 39, 32002])
        chunk_input_ids = input_processed["input_ids"].tensor_split(batch_dim)
        chunk_input_mask = input_processed["attention_mask"].tensor_split(batch_dim)
        chunk_prompt_lens = context_prompt_lens.tensor_split(batch_dim)

        res_logits = []
        for l, i, m, s in zip(chunk_logits, chunk_input_ids, chunk_input_mask, chunk_prompt_lens):
            res_logit = self.get_choices_logits(l, i, m, s)
            res_logits.append(res_logit)
        res_logits = torch.cat(res_logits, dim=0)

        #apply temperature
        res_logits = res_logits / temperature

        probs = Categorical(logits=res_logits.view(batch_dim, -1))
        if action is None:
            action = probs.sample()

        output["action"] = action
        output["log_prob"] = probs.log_prob(action)
        output["entropy"] = probs.entropy()

        return output

    # load models and model components
    def _load_model(self, checkpoint="microsoft/phi-2", is_lora=False, padding_side="right", gradient_ckpt=False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            #bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )

        # load the model first on the main process to download it if required.
        # the other processes will use the cached one
        with accelerator.main_process_first():
            config = AutoConfig.from_pretrained(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", trust_remote_code=True, quantization_config=quantization_config)

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_ckpt, gradient_checkpointing_kwargs={'use_reentrant': False})
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, padding_side=padding_side)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if is_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["k_proj", "v_proj", "q_proj"],
                # target_modules=r".*(perceiver_resampler|vision_model).*(q_proj|k_proj|v_proj)",
                lora_dropout=0.0,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters() 

        return model, tokenizer, config