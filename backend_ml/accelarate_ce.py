# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState, prepare_pippy


path = "/scratch/07980/sli4/data/gpt/meta-llama/Llama-2-7b-chat-hf/"
# sdpa implementation which is the default torch>2.1.2 fails with the tracing + attention mask kwarg
# with attn_implementation="eager" mode, the forward is very slow for some reason
model = AutoModelForCausalLM.from_pretrained(
    path, device_map="sequential", low_cpu_mem_usage=True, attn_implementation="sdpa"
)

model.eval()
print(model)

# Input configs
# Create example inputs for the model
tokenizer = AutoTokenizer.from_pretrained(path)
prompts = ("I would like to")  # bs = 1, sending 1 per process
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Create a pipeline stage from the model
# Using `auto` is equivalent to letting `device_map="auto"` figure
# out device mapping and will also split the model according to the
# number of total GPUs available if it fits on one GPU
model = prepare_pippy(model, split_points="auto", example_kwargs=inputs)

# You can pass `gather_output=True` to have the output from the model
# available on all GPUs
# model = prepare_pippy(model, split_points="auto", example_args=(input,), gather_output=True)

# currently we don't support `model.generate`
# output = model.generate(**inputs, max_new_tokens=1)
prompts = ("I would like to", "I really like to", "The weather is pretty")  # bs = 3
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
inputs = inputs.to(0)
with torch.no_grad():
    output = model(**inputs)

# The outputs are only on the final process by default
if PartialState().is_last_process:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
PartialState().destroy_process_group()