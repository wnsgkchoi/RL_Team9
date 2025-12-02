import torch
import torch.nn as nn
import numpy as np
import sys

from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

sys.path.append('<PATH_TO>/rocket')
sys.path.append('<PATH_TO>/rocket/src')

torch.set_default_device("cuda")

from src.agents.LLaVa15_agent import LLaVa15Agent


from enum import Enum
class CustomActions(Enum):
    move_left = "move west to reach the goal safely."
    move_down = "move south to reach the goal safely."
    move_right = "move east to reach the goal safely."
    move_up = "move north to reach the goal safely."

actions = [action.value for action in CustomActions]
print(actions)

agent = LLaVa15Agent("llava-hf/llava-1.5-7b-hf",
                            action_enum=CustomActions, 
                            is_lora=True,
                            padding_side="right", 
                            num_prompt_images=1, 
                            use_text_description=True, 
                            gradient_ckpt=False)
agent.critic.to('cuda')

#model, processor, config = load_model(checkpoint="llava-hf/llava-1.5-7b-hf", is_lora=True, padding_side="left")
print(f"{type(agent.processor)=}")
print(f"model config: {agent.config}")

fake_image = torch.randint(0,255,(2,3,244,244))
print(fake_image.shape)

#res = agent.get_action_and_value(fake_image, text_description=['this is a fake text description'], temperature=1.0)
#print(res)

import requests
from PIL import Image

image1 = Image.open("<PATH_TO>/frozen_starting_frame.png")
image2 = Image.open("<PATH_TO>/frozen2.png")
# image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
#image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
# image1.show()
# image.show()

prompts = [
            "USER: <image>\nYou are an agent in a survival 2D game as shown in this image.\nASSISTANT:",
            "USER: <image>\nWhat do you see in the image"
]

inputs = agent.processor(prompts, images=fake_image, padding=True, return_tensors="pt").to("cuda")
for k,v in inputs.items():
  print(k,v.shape, type(v))

print(f"{agent.processor.batch_decode(inputs.input_ids)=}")

res = agent.network(**inputs, output_hidden_states=True)
        # get the context len, useful to locate the action tokens when computing the logits
lens = torch.sum(inputs["attention_mask"], dim=1)
print(f"{res.keys()=}")
print(f"{res.logits.shape=}")
print(f"{len(res.hidden_states)=}")
print(f"{res.hidden_states[0].shape=}")
print(f"{lens=}")

shift_labels = inputs.input_ids[1, -10:]
shift_logits = res.logits[1, :, :]
most_likely_tokens = torch.argmax(shift_logits, dim=-1)
print(f"{most_likely_tokens=}")
print(f"most likely sequence:{agent.processor.batch_decode(most_likely_tokens, skip_special_tokens=False)=}")


# output = model.generate(**inputs, max_new_tokens=100)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# for text in generated_text:
#   print(text)
