#!/usr/bin/env python3
# =============================================================================
# Description:
# This script evaluates the correctness of model-generated mathematical 
# answers (e.g., from LLM completions) against gold solutions in a `.jsonl` file.
# 
# It supports two evaluation modes:
#   (1) `evaluate`: Applies majority, weighted, and mean-weighted voting over 
#       completions per problem to determine correctness and compute accuracy.
#   (2) `evaluate_per_completion_index`: Computes accuracy for each individual 
#       completion position to analyze variation in quality across outputs.
#
# The script uses symbolic math matching (via SymPy) and LaTeX-based answer
# parsing to assess equivalence between predicted and gold answers.
#
# Inputs:
# - A `.jsonl` file where each line contains:
#     • "completions": list of generated answers
#     • "agg_scores": list of confidence scores per completion
#     • "answer": the ground truth answer

# NOTE: Use this to Evaluate AMC and AIME datasets, the parser is not good for MATH500



INPUT_FILE ="path_to_generated_resposes.jsonl"
SCORE_THRESHOLD = 0.1 # completions with agg_score < threshold are ignored
VERBOSE         = True  # set False for quieter per-problem output
# ───────────────────────────────────


import json, sys
from pathlib import Path
from collections import Counter, defaultdict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from sympy import Eq, simplify, SympifyError

def verify_equal(pred_raw: str, gold_raw: str) -> bool:
    """
    Lightweight alternative to math_verify.verify().
    Returns True iff `pred_raw` and `gold_raw` represent the same value.
    Works for integers, rationals, and most elementary expressions.
    """
    try:
        # parse to SymPy
        p = simplify(_parsed(pred_raw)[-1])   # last match
        g = simplify(_parsed(gold_raw)[-1])
        return bool(Eq(p, g))
    except (IndexError, SympifyError):
        # fallback: numeric or case-insensitive string compare
        try:
            return float(pred_raw) == float(gold_raw)
        except ValueError:
            return pred_raw.strip().lower() == gold_raw
            # strip().lower()

# ───────────────────────────────────
# helper: parsing LaTeX answers
# ───────────────────────────────────
LATEX_CONF = [LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        nits=False, malformed_operators=False, basic_latex=True, boxed=True, units=True),
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)]

def _parsed(txt): 
    return parse(txt, extraction_config=LATEX_CONF)

def extract_final_answer(text: str) -> str:
    """
    Return a cleaned answer string:
      • strips \boxed{…} (even if followed by a period)
      • strips trailing punctuation
      • removes leading zeros in pure-integer answers
    """
    match = _parsed(text)
    if not match:
        return ""
    ans = str(match[-1]).rstrip(". ").strip()

    # strip \boxed{…}
    if ans.startswith(r"\boxed{"):
        # tolerate either "...}" or "...}."
        end = ans.find("}")
        if end != -1:
            ans = ans[len(r"\boxed{"):end]

    ans = ans.strip()

    # normalise integers:  025 → 25
    if ans.lstrip("-").isdigit():
        ans = str(int(ans))

    return ans


# ───────────────────────────────────
# voting utilities
# ───────────────────────────────────
def majority_vote_topk(ans_list, k=2):                        
    return [a for a, _ in Counter(ans_list).most_common(k)]   

def weighted_vote_topk(ans_list, weights, k=2):               
    bucket = defaultdict(float)                               
    for a, w in zip(ans_list, weights):                       
        bucket[a] += w                                        
    ranked = sorted(bucket.items(), key=lambda kv: -kv[1])    
    return [a for a, _ in ranked[:k]]                         

def mean_weighted_vote_topk(ans_list, weights, k=2):          
    bucket = defaultdict(list)                                
    for a, w in zip(ans_list, weights):                       
        bucket[a].append(w)                                   
    ranked = sorted(bucket.items(),                           
                    key=lambda kv: -(sum(kv[1]) / len(kv[1])))
    return [a for a, _ in ranked[:k]]                         


# ───────────────────────────────────
# evaluation loop 
# ───────────────────────────────────
def evaluate(path: Path, thr: float):

    # ---------------------------------------------
    # Function: evaluate
    # ---------------------------------------------
    # Evaluates the overall accuracy of generated completions against gold answers
    # by aggregating and verifying outputs using three voting strategies:
    #   - majority vote (most frequent answer)
    #   - weighted vote (by score)
    #   - mean-weighted vote (average score per answer)
    #
    # The function reads a `.jsonl` file where each line contains:
    #   - a gold answer
    #   - a list of generated completions
    #   - a corresponding list of scores
    #
    # It filters out low-confidence or empty completions, applies top-1 and top-2
    # voting strategies, checks equivalence with the gold answer using symbolic math
    # comparison, and reports summary statistics including accuracy for each method.
    #
    # Parameters:
    # - path (Path): Path to the `.jsonl` file containing completions and scores.
    # - thr (float): Minimum score threshold for a completion to be considered.

    summary = {
        "maj": 0, "wgt": 0, "avg": 0,         # top-1
        "maj2": 0, "wgt2": 0, "avg2": 0,      # top-2
        "n": 0
    }

    with path.open(encoding="utf-8") as fh:
        for idx, raw in enumerate(fh, 1):
            rec  = json.loads(raw)
            gold = rec["answer"]

            # ----------------------------------------------------------
            # keep completions ≥ threshold and with non-empty answers
            # ----------------------------------------------------------
            pairs = [[extract_final_answer(c), s]
                     for c, s in zip(rec["completions"], rec["agg_scores"])]
            
            # propagate answers inside each 2-completion group
            for i in range(0, len(pairs) - 1, 2):
                a0, s0 = pairs[i]
                a1, s1 = pairs[i + 1]
                if bool(a0) ^ bool(a1):
                    fill = a0 or a1
                    pairs[i][0] = fill
                    pairs[i + 1][0] = fill

            extracted = [(a, s) for a, s in pairs if a and s >= thr]

            # ----------------------------------------------------------
            # voting
            # ----------------------------------------------------------
            if extracted:
                finals, scores = zip(*extracted)

                maj_ans     = majority_vote_topk(finals, 1)[0]
                wgt_ans     = weighted_vote_topk(finals, scores, 1)[0]
                avg_ans     = mean_weighted_vote_topk(finals, scores, 1)[0]

                maj_top2    = majority_vote_topk(finals, 2)          # >>> NEW
                wgt_top2    = weighted_vote_topk(finals, scores, 2)  # >>> NEW
                avg_top2    = mean_weighted_vote_topk(finals, scores, 2)  # >>> NEW
            else:
                finals = scores = ()
                maj_ans = wgt_ans = avg_ans = ""
                maj_top2 = wgt_top2 = avg_top2 = []

            # ----------------------------------------------------------
            # correctness checks
            # ----------------------------------------------------------
            maj_ok  = verify_equal(maj_ans, gold) if maj_ans else False
            wgt_ok  = verify_equal(wgt_ans, gold) if wgt_ans else False
            avg_ok  = verify_equal(avg_ans, gold) if avg_ans else False

            maj2_ok = any(verify_equal(a, gold) for a in maj_top2)   # >>> NEW
            wgt2_ok = any(verify_equal(a, gold) for a in wgt_top2)   # >>> NEW
            avg2_ok = any(verify_equal(a, gold) for a in avg_top2)   # >>> NEW

            # ----------------------------------------------------------
            # update tallies
            # ----------------------------------------------------------
            summary["n"]   += 1
            summary["maj"] += maj_ok
            summary["wgt"] += wgt_ok
            summary["avg"] += avg_ok
            summary["maj2"] += maj2_ok                               # >>> NEW
            summary["wgt2"] += wgt2_ok                               # >>> NEW
            summary["avg2"] += avg2_ok                               # >>> NEW

            # 6) verbose reporting
            if not maj_ok or not wgt_ok:
                print(pairs)
                print(f"\nProblem {summary['n']} (jsonl line {idx})")
                if extracted:
                    print("  Extracted answers (after threshold & non-empty filter):")
                    for ans, sc in extracted:
                        print(f"    • {ans or '[NO ANSWER]':<30}  score={sc:.3f}")
                else:
                    print("  No completion met the score threshold.")
                print(f"  Gold            : {gold}")
                print(f"  Majority answer : {maj_ans or '[NONE]'}  -> {'✔' if maj_ok else '✘'}")
                print(f"  Weighted answer : {wgt_ans or '[NONE]'}  -> {'✔' if wgt_ok else '✘'}")
                print(f"  Mean-weighted   : {avg_ans or '[NONE]'}  -> {'✔' if avg_ok else '✘'}")
                print(f"  kept/total comps: {len(extracted)}/{len(rec['completions'])}")

    # ───────────────────────────────────
    # overall results
    # ───────────────────────────────────
    if summary["n"]:
        print("\n──────── summary ────────")
        print(f"Problems evaluated           : {summary['n']}")
        print(f"Majority-vote accuracy       : {summary['maj']/summary['n']:.3f}")
        print(f"Weighted-vote accuracy       : {summary['wgt']/summary['n']:.3f}")
        print(f"Mean-weighted accuracy       : {summary['avg']/summary['n']:.3f}")
        print("──────── top-2 (first OR second candidate correct) ────────")  # >>> NEW
        print(f"Majority top-2 accuracy      : {summary['maj2']/summary['n']:.3f}")  # >>> NEW
        print(f"Weighted top-2 accuracy      : {summary['wgt2']/summary['n']:.3f}")  # >>> NEW
        print(f"Mean-weighted top-2 accuracy : {summary['avg2']/summary['n']:.3f}")  # >>> NEW
    else:
        print("No problems processed.")


def evaluate_per_slot(path: Path, thr: float):
    # Evaluates multiple completions per example individually, reporting:
    #   - accuracy for each completion index (i.e., position in the list),
    #   - average accuracy across all completions.
    total_problems = 0
    correct_counts = None   # will become a list of length = #completions per problem

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            rec  = json.loads(raw)
            gold = rec["answer"]
            # extract all answers+scores once
            slots = [[extract_final_answer(c), s]
                     for c, s in zip(rec["completions"], rec["agg_scores"])]
            
            # propagate answers inside each 2-completion group
            for i in range(0, len(slots) - 1, 2):
                a0, s0 = slots[i]
                a1, s1 = slots[i + 1]
                if bool(a0) ^ bool(a1):
                    fill = a0 or a1
                    slots[i][0] = fill
                    slots[i + 1][0] = fill

            # initialize correct_counts on first pass
            if correct_counts is None:
                correct_counts = [0] * len(slots)

            # for this problem, check each slot j
            for j, (ans_j, score_j) in enumerate(slots):
                ok = False
                if ans_j and score_j >= thr:
                    ok = verify_equal(ans_j, gold)
                # increment if correct
                if ok:
                    correct_counts[j] += 1

            total_problems += 1

    # compute per‑slot accuracies
    accuracies = [cnt / total_problems for cnt in correct_counts]
    mean_acc  = sum(accuracies) / len(accuracies)

    # report
    print(f"Evaluated {total_problems} problems, each with {len(correct_counts)} slots")
    for j, acc in enumerate(accuracies):
        print(f"  Slot {j:2d} accuracy: {acc:.3f}")
    print(f"\nMean across slots: {mean_acc:.3f}")



# ───────────────────────────────────
# # run
# # ───────────────────────────────────
if __name__ == "__main__":
    fp = Path(INPUT_FILE)
    if not fp.exists():
        sys.exit(f"Input file not found: {fp}")
    evaluate(fp, SCORE_THRESHOLD)
    # evaluate_per_slot(fp, SCORE_THRESHOLD)








