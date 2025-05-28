#!/usr/bin/env python
# This file is a modification of code from:
#    “search-and-learn” (https://github.com/huggingface/search-and-learn)
# Modifed the original code to get the logprobs of generted tokens to be used to estimate the confidence

import copy
import logging
from dataclasses import dataclass

import numpy as np
from vllm import LLM, SamplingParams

from dataclasses import dataclass, field
from typing import Optional, List
# import torch

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
    force_token_added: bool = False 
    pruned: bool = False 
    completed: bool = False
    completion_tokens: int = 0
    history_embedding= None  
    next_embeddings = None


@dataclass
class GenResult:
    index: int
    initial_prompt_id: list[int]     # Already passed token IDs for the prompt
    initial_prompt: str
    first_step_text: str
    first_step_tokens: list[int]
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    lookahead_tokens: list[int] = field(default_factory=list)  # New field to accumulate token IDs
    logprobs: list[float] = field(default_factory=list)



def generate_k_steps(
    templated_convs,  # Already token ID sequences (list[int])
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> list[Beam]:
    gen_results = []
    # templated_convs now is a list of token ID lists.
    for i, conv_ids in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt_id=conv_ids,  # use provided token IDs
                initial_prompt="",  # optional: set to decoded text if needed
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason="",
                first_step_tokens=[],
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # Greedy for the rest of the steps
        # Process only those generations that did not finish with EOS
        current_gen = [
            gr for gr in gen_results
            if gr.stop_reason != "EOS"
        ]
        # Concatenate the already-passed token IDs with any generated tokens so far.
        gen_prompts = [
            gr.initial_prompt_id + gr.lookahead_tokens
            for gr in current_gen
        ]
        llm_outputs = llm.generate(
            prompt_token_ids=gen_prompts,
            sampling_params=gen_sampling_params,
            use_tqdm=False
        )
        for gr, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text

            # Extract the token_id and logprob from output of VLLM
            token_logprobs = output.outputs[0].logprobs
            temp_log_probs = []
            temp_tokens = []
            for step in token_logprobs:
                token_id, logprob_object = next(iter(step.items()))
                temp_tokens.append(token_id)
                temp_log_probs.append(logprob_object.logprob)

            gr.logprobs.extend(temp_log_probs)

            if i == 0:
                gr.first_step_tokens = temp_tokens
                gr.first_step_text = gen_text
                gr.first_step_stop_reason = output.outputs[0].stop_reason
                if gr.first_step_stop_reason is None:
                    gr.first_step_stop_reason = "EOS"

            # Update the generated tokens (using token IDs) and the text (for logging)
            gr.lookahead_tokens.extend(temp_tokens)
            gr.lookahead_text = gr.lookahead_text + gen_text

            gr.stop_reason = output.outputs[0].stop_reason
            if gr.stop_reason is None:
                gr.stop_reason = "EOS"

    outputs: list[Beam] = []
    counter = 0
    for i, conv_ids in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        all_logprobs = []  # For scoring later
        next_tokens = []
        for j in range(beam_width):
            gr = gen_results[counter]
            next_texts.append(gr.first_step_text)
            next_tokens.append(gr.first_step_tokens)
            lookahead_texts.append(gr.lookahead_text)
            stop_reasons.append(gr.first_step_stop_reason)
            all_logprobs.append(gr.logprobs)
            counter += 1

        beam_result = Beam(
            prompt="",             # Optionally, decode conv_ids if needed
            prompt_tokens=[],      # Fill as appropriate
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            log_probs=all_logprobs,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            next_tokens=next_tokens,
            current_tokens=[],
            previous_tokens=None,
        )
        outputs.append(beam_result)

    return outputs