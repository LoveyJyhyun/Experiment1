"""
load_dataset.py

Load benchmark datasets for hallucination evaluation experiments.
Supports: TruthfulQA, QAFactEval, HallucinationEval, CohS, and Custom CSV/JSON.
Outputs: Standardized format for generation pipeline.
"""

import os
import json
import argparse
import pandas as pd
from datasets import load_dataset

# -------- CONFIG -------- #
OUTPUT_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- HELPER FUNCTIONS -------- #

def save_as_json(data, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Saved: {path}")

# -------- DATASET LOADERS -------- #

def load_truthfulqa(split="validation", n_samples=100):
    ds = load_dataset("truthful_qa", "generation")[split]
    data = []
    for i, row in enumerate(ds.select(range(min(n_samples, len(ds))))):
        data.append({
            "dataset": "TruthfulQA",
            "question": row["question"],
            "reference_answer": row["best_answer"]
        })
    return data

def load_qafacteval(split="test", n_samples=100):
    ds = load_dataset("QAFactEval")[split]
    data = []
    for i, row in enumerate(ds.select(range(min(n_samples, len(ds))))):
        data.append({
            "dataset": "QAFactEval",
            "question": row["qa_pair"]["question"],
            "reference_answer": row["qa_pair"]["answer"]
        })
    return data

def load_hallucinationeval(split="test", n_samples=100):
    ds = load_dataset("hallucination_eval")[split]
    data = []
    for i, row in enumerate(ds.select(range(min(n_samples, len(ds))))):
        data.append({
            "dataset": "HallucinationEval",
            "question": row.get("question", "N/A"),
            "reference_answer": row.get("reference_answer", row.get("gold_answer", ""))
        })
    return data

def load_cohs(split="test", n_samples=100):
    ds = load_dataset("cais/multi-fact-summ-cohs")[split]
    data = []
    for i, row in enumerate(ds.select(range(min(n_samples, len(ds))))):
        data.append({
            "dataset": "CohS",
            "question": row.get("source", "")[:300],  # Use source text as pseudo-question
            "reference_answer": row.get("summary", "")
        })
    return data

def load_custom_file(file_path, format="csv", n_samples=100):
    if format == "csv":
        df = pd.read_csv(file_path)
    elif format == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format.")

    if "question" not in df.columns or "reference_answer" not in df.columns:
        raise ValueError("Custom file must contain 'question' and 'reference_answer' columns.")

    df = df.head(n_samples)
    data = df.to_dict(orient="records")
    return data

# -------- MAIN FUNCTION -------- #

def main():
    parser = argparse.ArgumentParser(description="Load dataset and format for generation pipeline")
    parser.add_argument("--dataset", required=True,
                        choices=["truthfulqa", "qafacteval", "hallucinationeval", "cohs", "custom"],
                        help="Dataset to load")
    parser.add_argument("--split", default="test", help="Dataset split (if available)")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to load")
    parser.add_argument("--custom_path", help="Path to custom dataset file")
    parser.add_argument("--custom_format", choices=["csv", "json"], help="Custom file format")

    args = parser.parse_args()

    if args.dataset == "truthfulqa":
        data = load_truthfulqa(split=args.split, n_samples=args.n_samples)
    elif args.dataset == "qafacteval":
        data = load_qafacteval(split=args.split, n_samples=args.n_samples)
    elif args.dataset == "hallucinationeval":
        data = load_hallucinationeval(split=args.split, n_samples=args.n_samples)
    elif args.dataset == "cohs":
        data = load_cohs(split=args.split, n_samples=args.n_samples)
    elif args.dataset == "custom":
        if not args.custom_path or not args.custom_format:
            raise ValueError("For custom dataset, both --custom_path and --custom_format are required.")
        data = load_custom_file(file_path=args.custom_path, format=args.custom_format, n_samples=args.n_samples)
    else:
        raise ValueError("Unsupported dataset.")

    save_as_json(data, f"{args.dataset}_{args.split}_{args.n_samples}")

if __name__ == "__main__":
    main()
