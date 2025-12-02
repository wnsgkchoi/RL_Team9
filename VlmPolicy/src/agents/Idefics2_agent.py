import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import copy

from thefuzz import fuzz, process
from string import punctuation
import re

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    AutoConfig,
    Idefics2ForConditionalGeneration
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from torch.distributions.categorical import Categorical
from accelerate import Accelerator

from agents.base_agent import BaseAgent

accelerator = Accelerator()

class Idefics2Agent(nn.Module, BaseAgent):
    def __init__(
        self, checkpoint_name, action_enum, is_lora=False, padding_side="right", num_prompt_images=1, use_text_description=True, gradient_ckpt=False, disable_adapters_for_generation=True, max_new_tokens=100
    ):
        super().__init__()
        self.action_enum = action_enum
        self.action_generation_cache = {}
        self.disable_adapters_for_generation = disable_adapters_for_generation
        self.is_lora = is_lora
        self.num_prompt_images = num_prompt_images
        self.use_text_description = use_text_description
        self.max_new_tokens = max_new_tokens
        self.network, self.processor, self.config = self._load_model(
            checkpoint_name, is_lora, padding_side, gradient_ckpt
        )
        # the critic remains in float32. (i.e. no lora or quantized stuff)
        self.critic = nn.Sequential(
            #nn.LazyLinear(1024),  # right now lazy leayers are not supported for distributed trainig
            nn.Linear(self.config.text_config.hidden_size, 1024),
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

    @torch.no_grad()
    def generate_continuation(self, prompt, frame):
        # generates a continuation of the given prompt based on the provided frame.
        model_inputs = self.processor(images=[frame], text=prompt, return_tensors="pt").to(self.network.device)
        prompt_len = model_inputs.input_ids.shape[-1]
        if self.disable_adapters_for_generation and self.is_lora:
            with self.network.disable_adapter():
                generated_ids = self.network.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)
        else:
            generated_ids = self.network.generate(**model_inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.processor.tokenizer.eos_token_id)
        generated_string = self.processor.decode(generated_ids[0][prompt_len:-1]) # [prompt_len:-1] selects only the generated sequence without the <end_of_utterance> appended by Idefics2
        return generated_string

    def find_action_in_sentence_and_create_action_choices(self, generated_action, action_list, fixtemplate_action_choices, advanced_action_matching=False):
        # extract the action from the generated_action.
        # substitute the action position with the other possible actions.
        # if no match is found return the defult action prompts.
        pattern = '|'.join(map(re.escape, action_list))
        #pattern = r'BEST ACTION:\s*(' + '|'.join(map(re.escape, action_list)) + r')\b'
        #pattern = r'ACTION:\s*(.*)'
        match = re.search(pattern, generated_action, flags=re.IGNORECASE)
        if match:
            match_found = 1
            matched_action = match.group()
            generated_action_choices = [generated_action.replace(matched_action, f'{action}') for action in action_list]
            return generated_action_choices, match_found
        
        # second case: search for actions with (the least number of) characters in between them: e.g. move ... north
        #processed_action_list = [r'[^\.]*?'.join(a.split(' ')) for a in action_list]
        #pattern = '|'.join(processed_action_list)
        #match = re.search(pattern, generated_action, flags=re.IGNORECASE)
        
        if advanced_action_matching and 'either' not in generated_action and match:  # if 'either' is present, we fall back to the default action prompts (since multiple actions are advised)
            match_found = 1
            matched_action_fuzzy = match.group()
            matched_action = process.extractOne(matched_action_fuzzy.lower().strip(), action_list, scorer=fuzz.token_set_ratio)[0]
            generated_action_choices = [] 
            for action in action_list:
                if action == matched_action:  # we use the matched action as is
                    generated_action_choices.append(generated_action)
                else:  # we replace the matched action with the other actions
                    start, end = match.span()
                    # we need to replace ONLY the matched occurrence
                    replaced_action = generated_action[:start] + action + generated_action[end:]
                    generated_action_choices.append(replaced_action)
            return generated_action_choices, match_found
        
        match_found = 0
        generated_action_choices = fixtemplate_action_choices
        return generated_action_choices, match_found
 
    def create_prompt_for_action_and_value(self, x, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None, generate_actions=False, advanced_action_matching=False):
        # text_description is None for the first step (after reset()) and if use_text_description is set to False.
        
        assert value_prompt_template.count("{}") == 1, "value_prompt_template does not contain one and just one instance of {}. check the config"
        assert action_template.count("{}") == 1, "action_template does not contain one and just one instance of {}. check the config"

        action_list = [action.value for action in self.action_enum]

        frames = [torchvision.transforms.functional.to_pil_image(o) for o in x]

        #create the value prompt injecting the state textual description within the value_prompt_template
        v_prompts = [(value_prompt_template.format(t) if t is not None else value_prompt_template.format('')) for t in text_description]
        assert len(v_prompts) == len(frames)

        prompts, just_context_prompts, actions_word_count = [], [], []
        for vp, frame in zip(v_prompts, frames):
            #conversation = [{"role": "user", "content": vp}]
            # in Idefics2 the conversation doesn't carry the images, but just placeholders, which will be filled by the processor.
            # TODO add the possibility to use more than one image
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": vp},
                ],
            }]

            chat_vp = self.processor.apply_chat_template(conversation, tokenize=False)
            chat_vp_and_frame = (chat_vp, frame)
            just_context_prompts.append(chat_vp_and_frame)

            # set the number of times the action has been found in the generated text to 0
            match_found = 0
            generated_action = ''
            # if we need just the value prompt there is no need for actions
            if not just_value:
                # create the actions with fixed template, which are also used in the case of generate_action as fallback when there is no action match
                fixtemplate_action_choices = [action_template.format(action) for action in action_list]
                if generate_actions:
                    # check if the the current game state has already been seen and we have the generated action_choices.
                    # action_choices = self.action_generation_cache.get(chat_vp, None)
                    # TODO fix the fact that the images are not taken into consideration in the cache.
                    cache_hit = self.action_generation_cache.get(chat_vp, None)
                    if cache_hit is None:
                        # generate the action choices
                        generated_action = self.generate_continuation(chat_vp_and_frame[0], chat_vp_and_frame[1])
                        action_choices, match_found = self.find_action_in_sentence_and_create_action_choices(generated_action, action_list, fixtemplate_action_choices, advanced_action_matching)
                        #if the adapters are not disabled the model is keeping changing, so the cache is not useful
                        if self.disable_adapters_for_generation: 
                            self.action_generation_cache[chat_vp] = (action_choices, generated_action) # (list(str), str)
                    else:
                        action_choices = cache_hit[0]
                else:
                    action_choices = fixtemplate_action_choices

                for action in action_choices:
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": vp},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"{action}"},
                        ],
                    }]
                    #v = chat_vp + f" {action}"
                    v = self.processor.apply_chat_template(conversation, tokenize=False)
                    v_and_frame = (v, frame)
                    prompts.append(v_and_frame)
                    word_count = len([s.strip(punctuation) for s in action.split(' ')])
                    actions_word_count.append(word_count)                    
            else:
                # we are here just for the value prompt (i.e. no actions)
                prompts.append(chat_vp_and_frame)

        return prompts, just_context_prompts, match_found, actions_word_count, generated_action

    def get_value(
        self, x, value_prompt_template='{}', text_description=None, generate_actions=False
    ):
        return self.get_action_and_value_one_forward(
            x,
            action=None,
            just_value=True,
            value_prompt_template=value_prompt_template,
            text_description=text_description,
            generate_actions=False,
        )

    def get_action_and_value(
        self,
        x,
        value_prompt_template='{}',
        action_template='{}',
        action=None,
        text_description=None,
        temperature=1.0,
        generate_actions=False,
        normalization_by_words=False,
        action_logits_from_whole_seq=False,
        advanced_action_matching=False
    ):
        return self.get_action_and_value_one_forward(
            x,
            action=action,
            just_value=False,
            value_prompt_template=value_prompt_template,
            action_template=action_template,
            text_description=text_description,
            temperature=temperature,
            generate_actions=generate_actions,
            normalization_by_words=normalization_by_words,
            action_logits_from_whole_seq=action_logits_from_whole_seq,
            advanced_action_matching=advanced_action_matching
        )

    def get_action_and_value_one_forward(self, x, action=None, just_value=False, value_prompt_template='{}', action_template='{}', text_description=None, temperature=1.0, generate_actions=False, normalization_by_words=False, action_logits_from_whole_seq=False, advanced_action_matching=False):
        output = {"actions": None, "log_prob": None, "entropy": None, "values": None}

        # step in common for actions and values computation
        batch_dim = len(text_description)
        x = self._preprocess_input(x)
        # print(f"{batch_dim=}")
        prompts, just_context_prompt, match_found, actions_word_count, generated_action = self.create_prompt_for_action_and_value(
                                                                                            x, 
                                                                                            just_value=just_value,
                                                                                            value_prompt_template=value_prompt_template,
                                                                                            action_template=action_template,
                                                                                            text_description=text_description,
                                                                                            generate_actions=generate_actions,
                                                                                            advanced_action_matching=advanced_action_matching)
        output["generated_action"] = generated_action
        output["prompts"] = prompts
        output["just_context_prompt"] = just_context_prompt

        context_prompt = self.processor(images=[[f] for _,f in just_context_prompt], text=[p for p,_ in just_context_prompt], return_tensors="pt", padding=True)
        input_processed = self.processor(images=[[f] for _,f in prompts], text=[p for p,_ in prompts], return_tensors="pt", padding=True).to(self.network.device)

        # get the context len, useful to locate the action tokens when computing the logits
        context_prompt_lens = torch.sum(context_prompt["attention_mask"], dim=1)

        padding_on_left_c = (context_prompt["attention_mask"][..., 0] == 0).sum() > (context_prompt["attention_mask"][..., -1] == 0).sum()
        padding_on_left_j = (input_processed["attention_mask"][..., 0] == 0).sum() > (input_processed["attention_mask"][..., -1] == 0).sum()

        assert not padding_on_left_c
        assert not padding_on_left_j

        result = self.network(**input_processed, output_hidden_states=True)

        if just_value:
            cls_token_hidden = result.hidden_states[-1][torch.arange(len(context_prompt_lens)), context_prompt_lens - 1]
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
        chunk_prompt_lens = torch.ones_like(context_prompt_lens).tensor_split(batch_dim) if action_logits_from_whole_seq else context_prompt_lens.tensor_split(batch_dim)
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
        output["action_match_found"] = match_found

        return output
    
    def _preprocess_input(self, x):
        # input is [num_env, 64,64,3], and must become [num_env, 3,64,64]
        x = x.permute(0, 3, 1, 2)
        #x = x.float() / 255.0  # Normalize input
        return x

    # load models and model components
    def _load_model(self, checkpoint="HuggingFaceM4/idefics2-8b-chatty", is_lora=False, padding_side="right", gradient_ckpt=False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )

        # load the model first on the main process to download it if required.
        # the other processes will use the cached one
        with accelerator.main_process_first():
            model = Idefics2ForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config
            )
        processor = AutoProcessor.from_pretrained(
                checkpoint,
                do_image_splitting=False,
                padding_side=padding_side
            )
        config = AutoConfig.from_pretrained(checkpoint)

        if is_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
                #use_dora=False,
                #init_lora_weights="gaussian"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters() 

        return model, processor, config
