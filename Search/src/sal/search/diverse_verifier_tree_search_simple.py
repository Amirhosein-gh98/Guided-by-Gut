# This file is a modification of code from:
#    “search-and-learn” (https://github.com/huggingface/search-and-learn)
#
# Modifications:
#  - Implements Diverse Verifier Tree Search routine.
#  - Allows switching between two search-guidance modes:
#      * If `prm` is provided, uses the external Process Reward Model (PRM)
#        to score and guide candidate expansions.
#      * Otherwise, falls back to our “Guided by Gut” (GG) strategy,
#        which uses intrinsic LLM signals (confidence & novelty) for self-guidance.
#  - In the paper this code is used to compare GG against PRMS (table2)


import logging
from collections import defaultdict
import re
import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sklearn.metrics.pairwise import cosine_similarity

import random
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict # Added Dict for type hint

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils_v4 import Beam, build_conv, generate_k_steps

logger = logging.getLogger()

import re
import string


def compute_beam_scores(
    beam: Beam,
    novelty_bonus: float = 0.1,
    gamma: float = 0.05) -> list[float]:
    known_tokens = set(beam.prompt_tokens) | set(beam.current_tokens)
    scores = []

    for next_texts, next_tokens, logprobs in zip(beam.next_texts, beam.next_tokens, beam.log_probs):
        if not logprobs or not next_texts.strip():
            scores.append(0.0)
            continue

        logprobs = np.exp(logprobs)
        mean_logprob = np.mean(logprobs)

        # 2. Token novelty: proportion of novel tokens
        if len(next_tokens) == 0:
            token_novelty = 0.0

        novel_count = sum(1 for token in next_tokens if token not in known_tokens)
        token_novelty = novel_count / len(next_tokens) * novelty_bonus


        # Combine scores
        final_score = mean_logprob + token_novelty 

        scores.append(final_score)
    return scores

def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm):


    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=2048,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        min_tokens=10,
        n=1,
        logprobs=1,
    )

    beams: list[Beam] = []
    tokenizer = llm.get_tokenizer()  # Assumed method to get the tokenizer


    # Create beams (each beam stores its own token IDs)
    for prompt in batch_of_prompts:
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        for i in range(config.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    index=i,
                    current_text="",  # you can still keep the decoded text for logging
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=[0.0],
                    log_probs=[],
                    all_scores=[],
                    previous_text=None,
                    history=[],
                    previous_tokens=None,
                    current_tokens=[],  # will store token IDs generated so far
                    next_tokens=None,
                    pruned=0,
                    completed=False,
                    completion_tokens=0,
                    force_token_added = False, 
                )
            )

    # Begin beam search iterations
    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        gen_beams = [b for b in beams if b.pruned!=2]
        if len(gen_beams) == 0:
            break

        # Adjust sampling parameters on final iteration if needed.
        if i == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=2048,
                top_p=config.top_p,
                n=1,
                logprobs=1,
            )


        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in gen_beams
        ]

        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_conv_ids = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=True,
        )

        # Call the modified generate_k_steps with token ID inputs
        gen_results = generate_k_steps(
            templated_conv_ids, 0, llm, sampling_params, config.beam_width
        )

        prompts_, completions_ = [], []
        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.next_tokens = gen_result.next_tokens
            beam.log_probs = gen_result.log_probs
            beam.stop_reasons = gen_result.stop_reasons

            prompts_.append(beam.prompt)
            completions_.append([beam.current_text + t for t in beam.next_texts])

        if prm:
            all_scores = prm.score(prompts_, completions_)

            for beam, scores in zip(gen_beams, all_scores, strict=True):
                agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores]
                best_score_ind = np.argmax(agg_scores)
                beam.all_scores = scores
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.history.append(beam.next_texts[best_score_ind])
                beam.best_scores = scores[best_score_ind]

                if (beam.next_texts[best_score_ind] == "" or beam.stop_reasons[best_score_ind] == "EOS"):
                    # stopped on EOS, prune
                    beam.pruned = 2
        else:
            # Score and select the best candidate for each beam
            for beam in gen_beams:
                agg_scores = compute_beam_scores(
                    beam,
                    novelty_bonus=config.novelty_bonus,
                    gamma=config.gamma
                )
                best_score_ind = np.argmax(agg_scores)
                beam.all_scores = [[score] for score in agg_scores]
                beam.best_scores = [agg_scores[best_score_ind]]

                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]

                if (beam.next_texts[best_score_ind] == "" or beam.stop_reasons[best_score_ind] == "EOS"):
                    # stopped on EOS, prune
                    beam.pruned = 2




        for beam in gen_beams:
            if "\\boxed{" in beam.next_texts[best_score_ind]:
                beam.pruned = 2
                    

    # Build final output beams (duplicating completions as needed)
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            if prm:
                score = beam.all_scores[i]
            else:
                score = [np.mean(np.exp(beam.log_probs[i]))]
            output.append(
                Beam(
                    prompt=beam.prompt,
                    prompt_tokens = beam.prompt_tokens,
                    index=beam.index,
                    current_text=beam.previous_text + beam.next_texts[i],
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores = score,
                    all_scores=beam.all_scores,
                    previous_text=beam.current_text,
                    pruned=beam.pruned,
                    history=[],
                    log_probs=beam.log_probs[i],
                    current_tokens = None,
                    previous_tokens = None,
                    next_tokens=None,
                )
            )
    return output


def dvts(examples, config: Config, llm: LLM, prm):
    problems = examples["problem"]
    beam_results = _dvts(problems, config, llm, prm)

    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(b.best_scores, config.agg_strategy)
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append(-1)

    return results