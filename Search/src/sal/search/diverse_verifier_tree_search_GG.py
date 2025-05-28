# This file is a modification of code from:
#    “search-and-learn” (https://github.com/huggingface/search-and-learn)
#
# Modifications:
#  - Main Implementation of Diverse Verifier Tree Search routine with self guidance.
#  - Uses intrinsic LLM signals (confidence & novelty) for self-guidance.



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


def check_consecutive_waits_windowed(
    text: str,
    window_size= 150) -> bool:

    # In the context of search-based reasoning, repetitive output tokens such as "wait" 
    # can signal degenerative or stalled generation. This utility function checks for 
    # excessive repetition of the token "wait" within a trailing window of text.
    # 
    # It is used as a lightweight degeneration detector during test-time search in 
    # Guided by Gut (GG), helping prune low-quality branches in the generation tree 
    # that exhibit repetition-based collapse.
    #
    # Parameters:
    # - text (str): The full generation output to analyze.
    # - window_size (int): The number of characters from the end of the text to consider 
    #                      (default: 150). If the text is shorter than this, the whole text is used.
    #
    # Returns:
    # - bool: True if "wait" appears more than 3 times in the window, indicating degeneration.
    if not isinstance(text, str) or not text:
        return False

    text_to_analyze = text
    if window_size is not None and window_size > 0 and window_size < len(text):
         text_to_analyze = text[-window_size:]


    lower_text_window = text_to_analyze.lower()

    cleaned_intermediate = re.sub(r'[^a-z\s]+', ' ', lower_text_window)

    words_in_window = cleaned_intermediate.split()
    count = words_in_window.count("wait")

    return count > 3


def compute_beam_scores(
    beam: Beam,
    itt,
    novelty_bonus: float = 0.1,
    total_itt: int = 200) -> list[float]:

    # Computes scores for each branch in a beam search during GG's intrinsic 
    # self-guided reasoning. The scoring integrates:
    #   - Confidence of the generated tokens (mean logprob),
    #   - Token novelty (how many tokens are new compared to the prompt/history),
    #   - Optional penalization for degenerate repetition using the "wait" checker,

    #
    # This scoring function aligns with GG’s strategy of replacing external PRM reward 
    # models with internal confidence and novelty signals. It is used within Diverse 
    # Verifier Tree Search (DVTS) to expand promising reasoning paths.
    #
    # Parameters:
    # - beam (Beam): An object encapsulating the current prompt, beam tokens, and candidates.
    # - itt (int): Current search iteration, used to gate novelty scoring and repetition checks.
    # - novelty_bonus (float): Scaling factor for novelty signal.
    # - total_itt (int): Total number of iterations allowed; helps restrict late-phase penalties.
    #
    # Returns:
    # - list[float]: Final score for each beam candidate, used to guide beam selection.

    known_tokens = set(beam.prompt_tokens) | set(beam.current_tokens)
    scores = []
    for next_texts, next_tokens, logprobs in zip(beam.next_texts, beam.next_tokens, beam.log_probs):
        if not logprobs:
            scores.append(0.0)
            continue

        logprobs = np.exp(logprobs)
        mean_logprob = np.mean(logprobs)

        # 2. Token novelty: proportion of novel tokens
        if len(next_tokens) == 0:
            token_novelty = 0.0

        if itt> 40 and itt%10 == 0 and itt < total_itt-3:
            if check_consecutive_waits_windowed(next_texts):
                token_novelty = -100
            else:
                novel_count = sum(1 for token in next_tokens if token not in known_tokens)
                token_novelty = novel_count / len(next_tokens) * novelty_bonus
        else:
            novel_count = sum(1 for token in next_tokens if token not in known_tokens)
            token_novelty = novel_count / len(next_tokens) * novelty_bonus


        final_score = mean_logprob + token_novelty 
        scores.append(final_score)
    return scores

def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        min_tokens=config.min_step_tokens,
        stop_token_ids=[382], #This is the token ID uses for splitting reasoning steps in Qwen Distill R1 models, change this based on the model
        include_stop_str_in_output=True,
        n=1,
        logprobs=1,
    )

    beams: list[Beam] = []
    tokenizer = llm.get_tokenizer()  # Assumed method to get the tokenizer

    # Pre-tokenize special markers
    if config.model_name_ == "Nemotron":
        begin_sentence_ids   = tokenizer.encode("<|im_start|>system\n<|im_end|>\n", add_special_tokens=False)
        user_tag_ids         = tokenizer.encode("<|im_start|>user\n",           add_special_tokens=False)
        assistant_tag_ids    = tokenizer.encode("<|im_start|>assistant\n",      add_special_tokens=False)
        end_tag_ids          = tokenizer.encode("<|im_end|>\n",                 add_special_tokens=False)
        dbl_newline_ids      = tokenizer.encode("\n\n", add_special_tokens=False)
        system_prompt_ids = tokenizer.encode(config.system_prompt, add_special_tokens=False)
    else:
        begin_sentence_ids = tokenizer.encode("<｜begin▁of▁sentence｜>", add_special_tokens=False)
        system_prompt_ids = tokenizer.encode(config.system_prompt, add_special_tokens=False)
        user_token_ids = tokenizer.encode("<｜User｜>", add_special_tokens=False)
        assistant_token_ids = tokenizer.encode("<｜Assistant｜>", add_special_tokens=False)

    if config.force_response:
        final_answer_marker_ids = tokenizer.encode(config.force_response, add_special_tokens=False)
    if config.check_prob_answer:
        check_prob_answer_ids = tokenizer.encode(config.check_prob_answer, add_special_tokens=False)

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
        if i <= 10:
            temperature = config.temperature0
        else:
            temperature = config.temperature
        sampling_params.temperature = temperature
        gen_beams = [b for b in beams if b.pruned!=2]
        if len(gen_beams) == 0:
            break


        if i == config.num_iterations - 2:
            temp_num_token = config.max_tokens - (config.num_iterations*(max(config.min_step_tokens, 40)))
            if temp_num_token > 100 and config.force_response !=None:
                sampling_params = SamplingParams(
                    temperature=config.temperature,
                    max_tokens = temp_num_token,
                    top_p=config.top_p,
                    n=1,
                    logprobs=1,
                )
        # Adjust sampling parameters on final iteration if needed.
        if i == config.num_iterations - 1:
            temp_num_token = max(5000, config.max_tokens - (config.num_iterations*(max(config.min_step_tokens, 100))))
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=temp_num_token,
                top_p=config.top_p,
                n=1,
                logprobs=1,
            )

        templated_conv_ids = [] 
        for b in gen_beams:
            if config.model_name_ == "Nemotron":
                input_ids = []
                input_ids.extend(begin_sentence_ids)           
                input_ids.extend(user_tag_ids)                  
                input_ids.extend(system_prompt_ids)             
                input_ids.extend(dbl_newline_ids)               
                input_ids.extend(b.prompt_tokens)               
                input_ids.extend(end_tag_ids) 
                input_ids.extend(assistant_tag_ids)            
            else:
                input_ids = []
                input_ids.extend(begin_sentence_ids)
                input_ids.extend(system_prompt_ids)
                input_ids.extend(user_token_ids)
                input_ids.extend(b.prompt_tokens)
                input_ids.extend(assistant_token_ids)
            # On subsequent iterations, add the generated tokens.
            if i != 0:
                if i == config.num_iterations - 1:
                    if config.force_response:
                        if config.force_response not in b.current_text:
                            b.current_tokens = b.current_tokens  + final_answer_marker_ids
                            b.current_text =  b.current_text + config.force_response
                input_ids.extend(b.current_tokens)
            templated_conv_ids.append(input_ids)

        # Call the modified generate_k_steps with token ID inputs, get the next steps and logprobs
        gen_results = generate_k_steps(
            templated_conv_ids, 0, llm, sampling_params, config.beam_width
        )

        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.next_tokens = gen_result.next_tokens
            beam.log_probs = gen_result.log_probs

        # Score and select the best candidate for each beam
        for beam in gen_beams:
            # print("beam:",beam.index)
            agg_scores = compute_beam_scores(
                beam,
                novelty_bonus=config.novelty_bonus,
                itt = i,
                total_itt = config.num_iterations
            )
            best_score_ind = np.argmax(agg_scores)
            beam.all_scores = [[score] for score in agg_scores]
            beam.best_scores = [agg_scores[best_score_ind]]

     
            if beam.best_scores[0] > -1:
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.previous_tokens = beam.current_tokens
                beam.current_tokens = beam.current_tokens + beam.next_tokens[best_score_ind]
            else:
                beam.previous_text = beam.current_text
                beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
                beam.previous_tokens = beam.current_tokens
                beam.current_tokens = beam.current_tokens + beam.next_tokens[best_score_ind]
                beam.pruned = 2

            if "\\boxed{" in beam.next_texts[best_score_ind]:
                beam.pruned = 2
            else:
                len_tokens_temp = len(beam.current_tokens)
                if len_tokens_temp > config.max_tokens:
                    if config.force_response not in beam.current_text:
                        beam.current_tokens = beam.current_tokens  + final_answer_marker_ids
                        beam.current_text =  beam.current_text + config.force_response
                    elif len_tokens_temp > config.max_tokens + 1000:
                        beam.pruned = 2
            

    # Build final output beams (duplicating completions as needed)
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            if beam.all_scores[i][0]< -10:
                score = [-1]
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
                    current_tokens = beam.previous_tokens + beam.next_tokens[i],
                    previous_tokens = beam.previous_tokens,
                    next_tokens=None,
                )
            )
    return output


def dvts4(examples, config: Config, llm: LLM, prm):
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