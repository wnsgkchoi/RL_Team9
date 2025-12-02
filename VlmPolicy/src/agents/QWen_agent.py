import torch
import torch.nn as nn
import numpy as np
import torchvision
import copy

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GPTQConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributions.categorical import Categorical
from accelerate import Accelerator
from string import punctuation

from agents.base_agent import BaseAgent

accelerator = Accelerator()

class QWenAgent(nn.Module, BaseAgent):
    def __init__(
        self, checkpoint_name, action_enum, is_lora=False, padding_side="left", num_prompt_images=1, use_text_description=False, gradient_ckpt=False
    ):
        super().__init__()
        # self.local_num_envs = local_num_envs
        self.num_prompt_images = num_prompt_images
        self.use_text_description = use_text_description
        self.is_lora = is_lora
        self.action_enum = action_enum

        self.network, self.tokenizer, self.config = self._load_model(
            checkpoint_name, is_lora, padding_side, gradient_ckpt
        )
        # the critic remains in float32. (i.e. no lora or quantized stuff)
        self.critic = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1024),  # Input layer
            #nn.Linear(self.config.vision_config.embed_dim, 1024),  # Input layer when the image embeddings are used as states
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
    
    def get_choices_logits(self, logits, labels, attention_mask, context_prompt_lens, actions_word_count, *, normalization_by_words=False):

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
        mean_tokens_logprobs_unmasked = (sum_unmasked / actions_word_count if normalization_by_words else
                                        sum_unmasked / count_unmasked)
        return mean_tokens_logprobs_unmasked

    def create_prompt_for_action_and_value(self, x, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None):
        # assume vector env right now, so x has dim [num_env, 3,244,244]
        # text_description is None for the first step (after reset()) and if use_text_description is set to False.

        text_obs = [f"{t}" if t is not None and self.use_text_description else "" for t in text_description]
        assert x.shape[0] == len(text_obs)

        fixtemplate_action_choices = [action_template.format(action.value) for action in self.action_enum]

        prompts, just_context_prompts, actions_word_count = [], [], []
        for t_obs in text_obs:
            v_prompt = value_prompt_template.format(t_obs)
            just_context_prompts.append(v_prompt)
            if not just_value:
                for action in fixtemplate_action_choices:
                    v = copy.deepcopy(v_prompt)
                    v = v + f"{action}"
                    prompts.append(v)
                    word_count = len([s.strip(punctuation) for s in action.split(' ')])
                    actions_word_count.append(word_count)
            else:
                prompts.append(v_prompt)
        return prompts, just_context_prompts, actions_word_count


    def get_value(self, x, value_prompt_template='{}', text_description=None, **kwargs):
        return self.get_action_and_value_one_forward(x, action=None, just_value=True, value_prompt_template=value_prompt_template, text_description=text_description)

    def get_action_and_value(self, x, value_prompt_template='{}', action_template='{}', action=None, text_description=None, temperature=1.0, normalization_by_words=False, action_logits_from_whole_seq=False, **kwargs):
        return self.get_action_and_value_one_forward(x, action=action, just_value=False, value_prompt_template=value_prompt_template, action_template=action_template, text_description=text_description, temperature=temperature, normalization_by_words=normalization_by_words, action_logits_from_whole_seq=action_logits_from_whole_seq)

    def get_action_and_value_one_forward(self, x, action=None, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None, temperature=1.0, normalization_by_words=False, action_logits_from_whole_seq=False):
        output = {"actions": None, "log_prob": None, "entropy": None, "values": None}

        # step in common for actions and values computation
        batch_dim = x.shape[0]
        action_number = len(self.action_enum)
        input_x = self._preprocess_input(x)

        prompts, just_context_prompts, actions_word_count = self.create_prompt_for_action_and_value(
            input_x, just_value=just_value, value_prompt_template=value_prompt_template, action_template=action_template, text_description=text_description
        )
        output["prompts"] = prompts
        output["just_context_prompts"] = just_context_prompts

        # check that every prompt has been replicated for every action and every different image observation
        if not just_value:
            assert len(prompts) == action_number * batch_dim, f"{action_number=}, {batch_dim=} != {len(prompts)=}\n the prompts are: {prompts}"
        else:
            assert len(prompts) == batch_dim, f"{batch_dim=} != {len(prompts)=}\n the prompts are: {prompts}"
            
        context_prompt = self.tokenizer(just_context_prompts, return_tensors='pt', padding=True).to(self.network.device)
        input_processed = self.tokenizer(prompts, return_tensors='pt', padding=True).to(self.network.device)

        # get the context len, useful to locate the action tokens when computing the logits
        context_prompt_lens = torch.sum(context_prompt["attention_mask"], dim=1)

        padding_on_left_c = (context_prompt["attention_mask"][..., 0] == 0).sum() > (context_prompt["attention_mask"][..., -1] == 0).sum()
        padding_on_left_j = (input_processed["attention_mask"][..., 0] == 0).sum() > (input_processed["attention_mask"][..., -1] == 0).sum()

        assert not padding_on_left_c
        assert not padding_on_left_j

        # replicate each image to align to the prompt replication.
        input_x = torch.repeat_interleave(input_x, action_number, dim=0)

        result = self.network(**input_processed, image_list=input_x, output_hidden_states=True)

        if just_value:
            cls_token_hidden = result.hidden_states[-1][torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]
            # sum the image hidden states to get the cls token embedding 
            #cls_token_hidden = result.image_hidden_states.sum(dim=2).squeeze(1) 
        else:
            cls_token_hidden_deduplicated = result.hidden_states[-1][::len(self.action_enum)] # select the 0th, 17th, 34th, ... rows
            cls_token_hidden = cls_token_hidden_deduplicated[torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]
            #cls_token_hidden_deduplicated = result.image_hidden_states[::len(self.action_enum)]
            #cls_token_hidden = cls_token_hidden_deduplicated.sum(dim=2).squeeze(1)

        values = self.critic(cls_token_hidden.float())
        output["values"] = values

        if just_value:
            return output

        # if more than 1 local_env we need to get_choices_logits for every one of them
        chunk_logits = result.logits.tensor_split(batch_dim)                              #[34, 39, 32002] -> #([17, 39, 32002], [17, 39, 32002])
        chunk_input_ids = input_processed["input_ids"].tensor_split(batch_dim)
        chunk_input_mask = input_processed["attention_mask"].tensor_split(batch_dim)
        #if action_logits_from_whole_seq is true, the context len becomes 1, so the action logits are computed on the entire sequence
        chunk_prompt_lens = context_prompt_lens.tensor_split(batch_dim) if action_logits_from_whole_seq else torch.ones_like(context_prompt_lens).tensor_split(batch_dim)
        chunk_actions_word_count = torch.tensor(actions_word_count, device=self.network.device).tensor_split(batch_dim)


        res_logits = []
        for l, i, m, s, c in zip(chunk_logits, chunk_input_ids, chunk_input_mask, chunk_prompt_lens, chunk_actions_word_count):
            res_logit = self.get_choices_logits(l, i, m, s, c, normalization_by_words=normalization_by_words)
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
        x = x.permute(0, 3, 1, 2).float()
        return x

    # load models and model components
    def _load_model(self, checkpoint, is_lora=False, padding_side="right", gradient_ckpt=False):

        quantization_config=GPTQConfig(
                bits=4, disable_exllama=True
            )
        # load the model first on the main process to download it if required.
        # the other processes will use the cached one

        config = AutoConfig.from_pretrained(
            checkpoint,
            #cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        config.use_cache = False

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            #cache_dir=training_args.cache_dir,
            device_map=None,
            trust_remote_code=True,
            quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            #cache_dir=training_args.cache_dir,
            model_max_length=2048,
            padding_side=padding_side,
            use_fast=False,
            trust_remote_code=True,
        )
        tokenizer.pad_token_id = tokenizer.eod_id
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_ckpt)#, gradient_checkpointing_kwargs={'use_reentrant': False})

        if is_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["c_attn", "attn.c_proj", "w1", "w2"],
                lora_dropout=0.0,
                bias="none",
                modules_to_save=None
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        if gradient_ckpt:
            model.enable_input_require_grads()

        return model, tokenizer, config
