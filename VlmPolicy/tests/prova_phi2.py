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
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

def load_model(checkpoint="microsoft/phi-2", is_lora=False, padding_side="left"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        config = AutoConfig.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", trust_remote_code=True, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if is_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],
                # target_modules=r".*(perceiver_resampler|vision_model).*(q_proj|k_proj|v_proj)",
                lora_dropout=0.0,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer, config

#model, tokenizer, config = load_model(checkpoint="microsoft/phi-2", is_lora=True, padding_side="left")
#print(type(tokenizer))
#print(f"model config: {config}")

#samples = ['Can you help me write a formal email to a potential business partner proposing a joint venture?', 'this is more than one input']

#inputs = tokenizer(samples, return_tensors="pt", padding=True)
#print(tokenizer.batch_decode(inputs.input_ids))

#res = model(**inputs, output_hidden_states=True)

#outputs = model.generate(**inputs, max_length=200)
#text = tokenizer.batch_decode(outputs)[0]
#print(text)
