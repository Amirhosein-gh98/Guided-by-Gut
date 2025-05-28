#!/usr/bin/env python
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

import logging
from vllm import LLM, SamplingParams
from sal.config import Config
from sal.models.reward_models import PRM

logger = logging.getLogger()

def generate_completions(prompts, config: Config, llm: LLM):
    """
    Generate a single completion for each prompt using the language model.
    
    Args:
        prompts (list[str]): List of input prompts.
        config (Config): Configuration object with generation parameters.
        llm (LLM): vLLM language model instance.
    
    Returns:
        list[str]: List of generated completions.
    """
    # Prepare input strings with system and user prompts
    # input_strs = [
    #     f"<|im_start|>system\n{config.system_prompt}<|im_end|>\n"
    #     f"<|im_start|>user\n{prompt}<|im_end|>\n"
    #     f"<|im_start|>assistant\n"
    #     for prompt in prompts
    # ]

    input_strs = [
    f"<｜begin▁of▁sentence｜>{config.system_prompt}<｜User｜>{prompt}<｜Assistant｜><think>\n"
    for prompt in prompts]

    # convs = [
    #     [
    #         {"role": "system",    "content": config.system_prompt},
    #         {"role": "user",      "content": prompt},
    #         {"role": "assistant", "content": None}
    #     ]
    #     for prompt in prompts
    # ]
    # tokenizer = llm.get_tokenizer()
    # if config.custom_chat_template is not None:
    #     tokenizer.chat_template = config.custom_chat_template

    # input_strs = [
    #     tokenizer.apply_chat_template(
    #         conv,
    #         tokenize=False,          
    #         add_generation_prompt=True
    #     )
    #     for conv in convs
    # ]

    # Define sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
    )
    # Generate completions using the LLM
    outputs = llm.generate(input_strs, sampling_params, use_tqdm=False)
    
    # Extract the generated text from each output
    completions = [output.outputs[0].text for output in outputs]
    return completions

def cot(examples, config: Config, llm: LLM, prm: PRM=None):
    """
    Process a batch of examples to generate completions using a simple CoT approach.
    
    Args:
        examples (dict): Dictionary containing a list of problems under the key "problem".
        config (Config): Configuration object with generation parameters.
        llm (LLM): vLLM language model instance.
    
    Returns:
        dict: Results containing completions, predictions, token counts, and empty scores.
    """
    # Extract the list of problems from examples
    problems = examples["problem"]
    
    # Generate one completion per prompt
    completions = generate_completions(problems, config, llm)
    
    # Construct the results dictionary
    results = {
        "completions": [[completion] for completion in completions],  # List of single-item lists
        "pred": completions,  # Same as completions since there's one per prompt
        "completion_tokens": [-1] * len(problems),  # Placeholder, as in original
        "scores": [[[1]] for _ in problems],  # Empty scores since scoring is removed
    }
    
    return results