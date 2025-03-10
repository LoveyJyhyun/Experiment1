"""
evaluate.py

Evaluates hallucination in generated outputs.
Supports: manual annotation loading or automated scoring baseline.
Outputs: hallucination scores per entry for attribution analysis.
"""

import os
import json
import argparse
import pandas as pd
from difflib import SequenceMatcher

# -------- CONFIGURATION -------- #
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- SCORING UTILS -------- #

def string_similarity(a, b):
    """Simple baseline scoring using text similarity"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def assign_hallucination_score(similarity, threshold_strong=0.85, threshold_weak=0.6):
    """
    Heuristic scoring based on similarity:
    0 = factual
    1 = weak mismatch (minor hallucination)
    2 = strong mismatch (major hallucination)
    """
    if similarity >= threshold_strong:
        return 0
    elif threshold_weak <= similarity < threshold_strong:
        return 1
    else:
        return 2

# -------- EVALUATION LOGIC -------- #

def evaluate_file(input_file, gold_file=None, strategy="similarity"):
    with open(input_file, "r") as f:
        data = json.load(f)

    if gold_file:
        with open(gold_file, "r") as f:
            gold_data = json.load(f)
        assert len(data) == len(gold_data), "Mismatch in input and gold data length"
    else:
        gold_data = None

    for i, item in enumerate(data):
        if strategy == "manual" and "hallucination_score" in item:
            continue  # Already annotated

        elif strategy == "similarity" and gold_data:
            reference = gold_data[i].get("reference_answer", "")
            similarity = string_similarity(item["response"], reference)
            item["similarity_score"] = round(similarity, 3)
            item["hallucination_score"] = assign_hallucination_score(similarity)

        else:
            # Default to zero if no gold standard exists (e.g., unsupervised run)
            item["hallucination_score"] = 0

    return data

# -------- MAIN SCRIPT -------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucinations in generated results")
    parser.add_argument("--input", required=True, help="Path to model response JSON file")
    parser.add_argument("--output", help="Path to save evaluated output (optional)")
    parser.add_argument("--gold", help="Path to gold/reference answers (optional)")
    parser.add_argument("--strategy", default="similarity", choices=["similarity", "manual"], help="Evaluation method")

    args = parser.parse_args()
    evaluated_data = evaluate_file(args.input, args.gold, args.strategy)

    # Output path
    output_path = args.output or args.input.replace(".json", "_scored.json")

    with open(output_path, "w") as f:
        json.dump(evaluated_data, f, indent=2)

    print(f"[✓] Evaluated file saved to: {output_path}")

if __name__ == "__main__":
    main()
