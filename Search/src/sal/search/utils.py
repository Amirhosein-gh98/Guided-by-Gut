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
import copy
import logging
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams

from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger()


def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)

@dataclass
class Beam:
    # Fields WITHOUT defaults first
    prompt: str
    prompt_tokens: list[int]
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]
    log_probs: list[list[float]]
    all_scores: list[list[float]]
    previous_text: str | None
    history: list[str] 
    previous_tokens: list[int] | None 
    current_tokens: list[int] | None 
    next_tokens: list[list[int]] | None 

    pruned: bool = False 
    completed: bool = False
    completion_tokens: int = 0

@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_tokens: list[int]
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    logprobs: list[float] = field(default_factory=list)
    
    


def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
                first_step_tokens=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text

            #Extracting the token_id and logprob from output of VLLM
            token_logprobs = output.outputs[0].logprobs
            temp_log_probs = []
            temp_tokens = []
            for step in token_logprobs:
                # This is likely the most direct and fastest way to get the single key-value pair
                token_id, logprob_object = next(iter(step.items()))
                temp_tokens.append(token_id)
            temp_log_probs.append(logprob_object.logprob)

            gen_result.logprobs.extend(temp_log_probs)

            if i == 0:
                gen_result.first_step_tokens = temp_tokens
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"




            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        all_logprobs = []  # Collect logprobs for scoring
        next_tokens = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            next_tokens.append(gen_result.first_step_tokens)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            all_logprobs.append(gen_result.logprobs)
            counter += 1

        beam_result = Beam(
            prompt=text,
            prompt_tokens=[],
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            log_probs = all_logprobs,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            next_tokens = next_tokens,
            current_tokens = [],
            previous_tokens = None,
        )
        outputs.append(beam_result)

    return outputs



# def generate_k_steps(
#     templated_convs: List[str],
#     beam_width: int,
#     lookahead_steps: int,
#     llm,
#     gen_sampling_params,
#     tokenizer,
#     embedding_layer,
#     score_threshold: float
# ) -> List[Beam]:
#     """
#     Optimized generation of next texts for each prompt, generating one child first and
#     additional children only if the first child's score is below the threshold.

#     Args:
#         templated_convs: List of input prompts.
#         beam_width: Number of children to generate per prompt if threshold not met.
#         lookahead_steps: Number of steps to look ahead for each child.
#         llm: Language model instance with a generate method.
#         gen_sampling_params: Parameters for generation (e.g., temperature, top_k).
#         tokenizer: Tokenizer for scoring.
#         embedding_layer: Embedding layer for scoring.
#         score_threshold: Threshold to decide if one child is sufficient.

#     Returns:
#         List of Beam objects, each containing next_texts (1 or beam_width) for a prompt.
#     """
#     # Step 1: Generate one child per prompt
#     gen_results = []
#     for i, text in enumerate(templated_convs):
#         gen_result = GenResult(
#             index=i,
#             initial_prompt=text,
#             first_step_text="",
#             lookahead_text="",
#             stop_reason=None,
#             first_step_stop_reason=None,
#             logprobs=[],
#         )
#         gen_results.append(gen_result)

#     # Generate lookahead for the first child of each prompt
#     for step in range(lookahead_steps + 1):
#         current_gen = [gen for gen in gen_results if gen.stop_reason != "EOS"]
#         if not current_gen:
#             break
#         gen_prompts = [gen.initial_prompt + gen.lookahead_text for gen in current_gen]
#         llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        
#         # Process outputs (simplified; adapt to your LLM output format)
#         for gen, output in zip(current_gen, llm_outputs):
#             new_text = output.text  # Adjust based on actual output structure
#             new_logprobs = output.logprobs  # Adjust based on actual output structure
#             gen.logprobs.extend(new_logprobs)
#             gen.lookahead_text += new_text
#             if step == 0:
#                 gen.first_step_text = new_text
#                 gen.first_step_stop_reason = output.stop_reason if output.stop_reason else None
#             if output.stop_reason == "EOS":
#                 gen.stop_reason = "EOS"

#     # Step 2: Score the first child for each prompt
#     scores = []
#     for gen_result in gen_results:
#         beam = Beam(
#             prompt=gen_result.initial_prompt,
#             index=gen_result.index,
#             current_text="",
#             next_texts=[gen_result.first_step_text],
#             lookahead_texts=[gen_result.lookahead_text],
#             stop_reasons=[gen_result.first_step_stop_reason],
#             log_probs=[gen_result.logprobs],
#         )
#         score = compute_beam_scores(beam, tokenizer, embedding_layer)[0]  # First score
#         scores.append(score)

#     # Step 3: Identify prompts needing more children
#     unsatisfied_indices = [i for i, score in enumerate(scores) if score <= score_threshold]

#     # Step 4: Generate additional children for unsatisfied prompts
#     additional_gen_results = []
#     for index in unsatisfied_indices:
#         text = templated_convs[index]
#         for j in range(beam_width - 1):  # -1 since one child is already generated
#             gen_result = GenResult(
#                 index=index,
#                 initial_prompt=text,
#                 first_step_text="",
#                 lookahead_text="",
#                 stop_reason=None,
#                 first_step_stop_reason=None,
#                 logprobs=[],
#             )
#             additional_gen_results.append(gen_result)

#     # Generate lookahead for additional children
#     for step in range(lookahead_steps + 1):
#         current_gen = [gen for gen in additional_gen_results if gen.stop_reason != "EOS"]
#         if not current_gen:
#             break
#         gen_prompts = [gen.initial_prompt + gen.lookahead_text for gen in current_gen]
#         llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        
#         # Process outputs
#         for gen, output in zip(current_gen, llm_outputs):
#             new_text = output.text
#             new_logprobs = output.logprobs
#             gen.logprobs.extend(new_logprobs)
#             gen.lookahead_text += new_text
#             if step == 0:
#                 gen.first_step_text = new_text
#                 gen.first_step_stop_reason = output.stop_reason if output.stop_reason else None
#             if output.stop_reason == "EOS":
#                 gen.stop_reason = "EOS"

#     # Step 5: Combine all results and create Beam objects
#     all_gen_results = gen_results + additional_gen_results
#     outputs = []
#     for i, text in enumerate(templated_convs):
#         beam_gen_results = [gen for gen in all_gen_results if gen.index == i]
#         next_texts = [gen.first_step_text for gen in beam_gen_results]
#         lookahead_texts = [gen.lookahead_text for gen in beam_gen_results]
#         stop_reasons = [gen.first_step_stop_reason for gen in beam_gen_results]
#         log_probs = [gen.logprobs for gen in beam_gen_results]
#         beam = Beam(
#             prompt=text,
#             index=i,
#             current_text="",
#             next_texts=next_texts,
#             lookahead_texts=lookahead_texts,
#             stop_reasons=stop_reasons,
#             log_probs=log_probs,
#         )
#         outputs.append(beam)

#     return outputs

# def compute_beam_scores(
#     beam: Beam,
#     tokenizer,
#     embedding_layer: torch.nn.Module,
#     novelty_bonus: float = 0.1,
#     gamma: float = 0.05,
#     lambda_: float = 0.2
# ) -> list[float]:
#     # Tokenize prompt and current_text to get known tokens
#     prompt_tokens = tokenizer.encode(beam.prompt, add_special_tokens=False)
#     current_text_tokens = tokenizer.encode(beam.current_text, add_special_tokens=False)
#     known_tokens = set(prompt_tokens) | set(current_text_tokens)

#     # Tokenize current_text and get embeddings
#     current_token_ids = tokenizer.encode(beam.current_text, add_special_tokens=False)
#     current_token_ids = torch.tensor(current_token_ids).unsqueeze(0)
#     # with torch.no_grad():
#     #     current_embeddings = embedding_layer(current_token_ids).mean(dim=1).cpu().numpy()

#     scores = []
#     for next_text, logprobs in zip(beam.next_texts, beam.log_probs):
#         if not logprobs:
#             scores.append(0.0)
#             continue

#         # 1. Mean log probability
#         mean_logprob = np.mean(logprobs)

#         # 2. Token novelty: proportion of novel tokens
#         next_text_tokens = tokenizer.encode(next_text, add_special_tokens=False)
#         if len(next_text_tokens) == 0:
#             token_novelty = 0.0
#         else:
#             novel_count = sum(1 for token in next_text_tokens if token not in known_tokens)
#             token_novelty = novel_count / len(next_text_tokens) * novelty_bonus

#         # 3. Uncertainty: variance of logprobs
#         uncertainty = np.var(logprobs) if len(logprobs) > 1 else 0.0


#         # Combine scores: sum of all four metrics
#         final_score = mean_logprob + token_novelty + gamma * uncertainty 
#         # + lambda_ * semantic_novelty
#         scores.append(final_score)

#     return scores