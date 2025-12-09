import torch
import torch.nn as nn
import numpy as np
import torchvision
import copy

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    IdeficsForVisionText2Text,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from torch.distributions.categorical import Categorical
from accelerate import Accelerator

from agents.base_agent import BaseAgent

# accelerator = Accelerator() # Removed global accelerator

class IDEFICSAgent(nn.Module, BaseAgent):
    def __init__(
        self, checkpoint_name, action_enum, is_lora=False, padding_side="left", num_prompt_images=1, use_text_description=False, gradient_ckpt=False, accelerator=None
    ):
        super().__init__()
        self.accelerator = accelerator
        # self.local_num_envs = local_num_envs
        self.num_prompt_images = num_prompt_images
        self.use_text_description = use_text_description
        self.is_lora = is_lora
        self.action_enum = action_enum

        self.network, self.processor, self.config = self._load_model(
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

    def get_choices_logits(self, logits, labels, attention_mask, context_prompt_lens):

        # select only actions
        logits = logits[..., context_prompt_lens - 1 :, :]
        attention_mask = attention_mask[..., context_prompt_lens - 1 :]
        labels = labels[..., context_prompt_lens - 1 :]

        shift_logits = logits[..., :-1, :]
        shift_attention = attention_mask[..., 1:]
        shift_labels = labels[..., 1:]

        # Safety check for gather to prevent CUDA device-side assert
        vocab_size = shift_logits.size(-1)
        if (shift_labels >= vocab_size).any() or (shift_labels < 0).any():
            print(f"[Error] Invalid token IDs in shift_labels!")
            print(f"Vocab size: {vocab_size}")
            print(f"Max label: {shift_labels.max().item()}")
            print(f"Min label: {shift_labels.min().item()}")
            # Force clamp to prevent crash and allow debugging
            shift_labels = torch.clamp(shift_labels, min=0, max=vocab_size-1)

        tokens_logprobs = torch.gather(
            shift_logits, 2, shift_labels[:, :, None]
        ).squeeze(-1)
        tokens_logprobs_applied_mask = tokens_logprobs * shift_attention

        # Calculate the sum and count of unmasked elements
        sum_unmasked = tokens_logprobs_applied_mask.sum(dim=1)
        count_unmasked = shift_attention.sum(dim=1)

        # Safety check for division by zero
        count_unmasked = torch.clamp(count_unmasked, min=1e-9)

        # Calculate the mean
        mean_tokens_logprobs_unmasked = sum_unmasked / count_unmasked
        return mean_tokens_logprobs_unmasked

    def create_prompt_for_action_and_value(self, x, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None):
        # assume vector env right now, so x has dim [num_env, 3,244,244]
        # text_description is None for the first step (after reset()) and if use_text_description is set to False.

        text_obs = [f"{t} " if t is not None and self.use_text_description else "" for t in text_description]
        assert x.shape[0] == len(text_obs)

        frames = [torchvision.transforms.functional.to_pil_image(o) for o in x]

        prompts, v_prompts = [], []
        for pil_image, t_obs in zip(frames, text_obs):
            if self.num_prompt_images == 1:
                v_prompt = [
                    "User:",
                    pil_image,
                    value_prompt_template.format(t_obs)
                ]
            else:
                v_prompt = [
                    value_prompt_template.format(t_obs)
                ]
            v_prompts.append(v_prompt)
            if not just_value:
                for action in self.action_enum:
                    v = copy.deepcopy(v_prompt)
                    p = [action_template.format(action.value)]
                    v[-1] = v[-1] + p[0]
                    prompts.append(v)
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
            input_x, just_value=just_value, value_prompt_template=value_prompt_template, action_template=action_template, text_description=text_description
        )
        output["prompts"] = prompts
        output["just_context_prompt"] = just_context_prompt

        #print(f"{output['prompts']=}")
        #print(f"{output['just_context_prompt']=}")

        # context_prompt = self.processor(just_context_prompt, add_end_of_utterance_token=False, return_tensors="pt", padding=True)
        # input_processed = self.processor(prompts, add_end_of_utterance_token=False, return_tensors="pt", padding=True).to(self.network.device)

        context_prompt = self.processor(text=just_context_prompt, add_end_of_utterance_token=False, return_tensors="pt", padding=True)
        input_processed = self.processor(text=prompts, add_end_of_utterance_token=False, return_tensors="pt", padding=True).to(self.network.device)

        # manually add padding masking as right padding doesn't add it
        context_prompt["attention_mask"][context_prompt["input_ids"] == self.processor.tokenizer.pad_token_id] = 0
        input_processed["attention_mask"][input_processed["input_ids"] == self.processor.tokenizer.pad_token_id] = 0

        # get the context len, useful to locate the action tokens when computing the logits
        context_prompt_lens = torch.sum(context_prompt["attention_mask"], dim=1)

        # padding_on_left_c = (context_prompt["attention_mask"][..., 0] == 0).sum() > (context_prompt["attention_mask"][..., -1] == 0).sum()
        # padding_on_left_j = (input_processed["attention_mask"][..., 0] == 0).sum() > (input_processed["attention_mask"][..., -1] == 0).sum()

        # assert not padding_on_left_c
        # assert not padding_on_left_j

        result = self.network(**input_processed, output_hidden_states=True)

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

        # Safety check for NaN/Inf in logits
        if torch.isnan(res_logits).any() or torch.isinf(res_logits).any():
            # print(f"[Warning] NaN or Inf detected in res_logits! Replacing with -1e9.")
            res_logits = torch.nan_to_num(res_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        
        # Ensure at least one value is finite and reasonable to prevent all -Inf
        # If a row is all -1e9 (was all -Inf), softmax will be uniform.
        # But to be safe, we can set the first element to 0 if all are small.
        # (Not strictly necessary if -1e9 is used, as exp(-1e9) is small but positive)

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

    # load models and model components
    def _load_model(self, checkpoint, is_lora=False, padding_side="left", gradient_ckpt=False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        
        # Determine device map
        device_map = None
        if self.accelerator is not None:
            device_map = {"": f"cuda:{self.accelerator.local_process_index}"}

        # load the model first on the main process to download it if required.
        # the other processes will use the cached one
        # Use self.accelerator if available, otherwise fallback (though it should be provided)
        if self.accelerator:
            context = self.accelerator.main_process_first()
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            config = AutoConfig.from_pretrained(checkpoint)
            model = IdeficsForVisionText2Text.from_pretrained(
                checkpoint, quantization_config=quantization_config, device_map=device_map
            )
            processor = AutoProcessor.from_pretrained(checkpoint, padding_side=padding_side)
        
        #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_ckpt, gradient_checkpointing_kwargs={'use_reentrant': False})

        if is_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj"],
                # target_modules=r".*(perceiver_resampler|vision_model).*(q_proj|k_proj|v_proj)",
                lora_dropout=0.0,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        processor.tokenizer.mask_token = "[mask]"
        processor.tokenizer.sep_token = "[sep]"
        processor.tokenizer.cls_token = "[cls]"
        # processor.tokenizer.cls_token_id = 25932 # this is the id for 'cls' (without angular brackets)
        processor.tokenizer.cls_token_id = 3158  # this is the id for 'action'

        return model, processor, config
