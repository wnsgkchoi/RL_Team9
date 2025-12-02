import torch
import torch.nn as nn
import numpy as np
import torchvision
import copy

from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from torch.distributions.categorical import Categorical
from accelerate import Accelerator

from agents.base_agent import BaseAgent

accelerator = Accelerator()

class LLaVa15Agent(nn.Module, BaseAgent):
    def __init__(
        self, checkpoint_name, action_enum, is_lora=False, padding_side="left", num_prompt_images=1, use_text_description=False, gradient_ckpt=False
    ):
        super().__init__()
        # self.local_num_envs = local_num_envs
        self.num_prompt_images = num_prompt_images
        self.use_text_description = use_text_description
        self.is_lora = is_lora
        self.action_enum = action_enum

        self.network, self.processor, self.config = self._load_model(
            checkpoint_name, is_lora, padding_side, gradient_ckpt
        )

        if hasattr(self.config.text_config, "hidden_size"):
            # llava 7b has the hidden_size but the 13b version doesn't
            hidden_dimension = self.config.text_config.hidden_size
        else:
            hidden_dimension = self.config.vision_config.intermediate_size
        
        # the critic remains in float32. (i.e. no lora or quantized stuff)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dimension, 1024),
            nn.ReLU(),  # ReLU activation function
            nn.Linear(1024, 512),  # Hidden layer 1
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, 1),  # Output layer
        )

    def get_lora_module_names(self):
        if self.is_lora:
            names = self.network.targeted_module_names
            return "\n".join(names)
        else:
            return 'No LORA'

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

    def create_prompt_for_action_and_value(self, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None):
        # assume vector env right now, so x has dim [num_env, 3,244,244]
        # text_description is None for the first step (after reset()) and if use_text_description is set to False.

        text_obs = [f"{t} " if t is not None and self.use_text_description else "" for t in text_description]

        prompts, v_prompts = [], []
        for t_obs in text_obs:
            image_tokens = ' ' + '<image>'*self.num_prompt_images
            v_prompt = f"USER:{image_tokens}\n{value_prompt_template.format(t_obs)}\nASSISTANT:"
        
            v_prompts.append(v_prompt)
            if not just_value:
                for action in self.action_enum:
                    v = copy.deepcopy(v_prompt)
                    p = v + f"{action_template.format(action.value)}"
                    prompts.append(p)
            else:
                prompts.append(v_prompt)
        return prompts, v_prompts

    def get_value(self, x, value_prompt_template='{}', text_description=None, **kwargs):
        return self.get_action_and_value_one_forward(x, action=None, just_value=True, value_prompt_template=value_prompt_template, text_description=text_description)

    def get_action_and_value(self, x, value_prompt_template='{}', action_template='{}', action=None, text_description=None, temperature=1.0, **kwargs):
        return self.get_action_and_value_one_forward(x, action=action, just_value=False, value_prompt_template=value_prompt_template, action_template=action_template, text_description=text_description, temperature=temperature)

    def get_action_and_value_one_forward(self, x, action=None, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None, temperature=1.0):
        output = {"actions": None, "log_prob": None, "entropy": None, "values": None}

        # step in common for actions and values computation
        batch_dim = x.shape[0]
        input_x = self._preprocess_input(x)

        prompts, just_context_prompt = self.create_prompt_for_action_and_value(
            just_value=just_value,
            value_prompt_template=value_prompt_template,
            action_template=action_template,
            text_description=text_description
        )
        output["prompts"] = prompts
        output["just_context_prompt"] = just_context_prompt

        frames = [torchvision.transforms.functional.to_pil_image(o) for o in input_x]

        if just_value:
            # prompts and just contexts are equal
            assert len(prompts) == batch_dim == len(just_context_prompt)
            frames_repeated = frames
        else:
            action_num = int(len(prompts) / batch_dim)
            assert action_num == len(self.action_enum), f"{action_num=} in not equal to {len(self.action_enum)=}, {batch_dim=}"
            # repeat the images action_num times to add them to the prompts
            frames_repeated = [f for f in frames for _ in range(action_num)] # [I1, I1, I1, ..., I2, I2, I2,...]

        context_prompt = self.processor(just_context_prompt, images=frames, padding=True, return_tensors="pt")
        input_processed = self.processor(prompts, images=frames_repeated, padding=True, return_tensors="pt").to(self.network.device)

        # manually add padding masking as right padding doesn't add it
        context_prompt["attention_mask"][context_prompt["input_ids"] == self.processor.tokenizer.pad_token_id] = 0
        input_processed["attention_mask"][input_processed["input_ids"] == self.processor.tokenizer.pad_token_id] = 0

        # get the context len, useful to locate the action tokens when computing the logits
        context_prompt_lens = torch.sum(context_prompt["attention_mask"], dim=1)

        padding_on_left_c = (context_prompt["attention_mask"][..., 0] == 0).sum() > (context_prompt["attention_mask"][..., -1] == 0).sum()
        padding_on_left_j = (input_processed["attention_mask"][..., 0] == 0).sum() > (input_processed["attention_mask"][..., -1] == 0).sum()

        assert not padding_on_left_c
        assert not padding_on_left_j

        result = self.network(**input_processed, output_hidden_states=True)

        prompts_max_len = input_processed["input_ids"].shape[-1]

        if just_value:
            # llava has the images tokens within the result.logits, hiddes_states...
            # we need to remove them first.
            final_hidden_layer_without_image_tokens = result.hidden_states[-1][:, -prompts_max_len:, :]
            cls_token_hidden = final_hidden_layer_without_image_tokens[torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]
        else:
            final_hidden_layer_without_image_tokens_deduplicated = result.hidden_states[-1][::len(self.action_enum), -prompts_max_len:, :]  # select the 0th, n_action_th, 2*n_actions_th, ... rows
            cls_token_hidden = final_hidden_layer_without_image_tokens_deduplicated[torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]

        values = self.critic(cls_token_hidden.float())
        output["values"] = values

        if just_value:
            return output

        # if more than 1 local_env we need to get_choices_logits for every one of them
        chunk_logits = result.logits[:, -prompts_max_len:, :].tensor_split(batch_dim)                              #[34, 39, 32002] -> #([17, 39, 32002], [17, 39, 32002])
        chunk_input_ids = input_processed["input_ids"].tensor_split(batch_dim)
        chunk_input_mask = input_processed["attention_mask"].tensor_split(batch_dim)
        chunk_prompt_lens = context_prompt_lens.tensor_split(batch_dim)

        res_logits = []
        for l, i, m, s in zip(chunk_logits, chunk_input_ids, chunk_input_mask, chunk_prompt_lens):
            res_logit = self.get_choices_logits(l, i, m, s)
            res_logits.append(res_logit)
        res_logits = torch.cat(res_logits, dim=0).view(batch_dim, -1)

        # apply temperature
        if temperature == 'max_logit':
            res_logits = res_logits - res_logits.max(-1, keepdim=True).values
        elif isinstance(temperature, float):
            res_logits = res_logits / temperature
        else:
            raise Exception(f"Invalid temperature: {temperature}")

        probs = Categorical(logits=res_logits)
        if action is None:
            action = probs.sample()

        output["action"] = action
        output["log_prob"] = probs.log_prob(action)
        output["entropy"] = probs.entropy()
        output["action_logits"] = probs.logits

        return output

    def _preprocess_input(self, x):
        # input is [num_env, 64,64,3], and must become [num_env, 3,64,64]
        x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0  # Normalize input
        return x

    def _load_model(self, checkpoint="llava-hf/llava-1.5-7b-hf", is_lora=False, padding_side="left", gradient_ckpt=False):
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        with accelerator.main_process_first():
            config = AutoConfig.from_pretrained(checkpoint)
            processor = AutoProcessor.from_pretrained(checkpoint, padding_side=padding_side)
            model = LlavaForConditionalGeneration.from_pretrained(
                checkpoint,
                quantization_config=quantization_config,
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_ckpt, gradient_checkpointing_kwargs={'use_reentrant': False})

        if is_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "v_proj",
                    "q_proj",
                    "up_proj",
                    "k_proj",
                ],
                lora_dropout=0.0,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, processor, config
