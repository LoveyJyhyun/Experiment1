"""
attribution_analysis.py

Compute Hallucination Rate (HR), Prompt Sensitivity (PS), and Model Variability (MV)
from generation + evaluation results of LLM hallucination experiments.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- CONFIGURATION -------- #
RESULTS_DIR = "results/"
OUTPUT_METRICS_FILE = "results/attribution_metrics.csv"
PLOT_DIR = "results/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------- LOAD RESULTS -------- #
def load_all_results(results_dir):
    all_data = []
    files = glob.glob(os.path.join(results_dir, "*.json"))
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            for entry in data:
                all_data.append(entry)
    return pd.DataFrame(all_data)

# -------- COMPUTE METRICS -------- #
def compute_hallucination_rate(df):
    """
    Assumes 'hallucination_score' is annotated:
    0 = factual, 1 = minor hallucination, 2 = major hallucination
    """
    df["is_hallucinated"] = df["hallucination_score"] >= 1
    hr = df.groupby(["model", "prompt_type"])["is_hallucinated"].mean().reset_index()
    hr.rename(columns={"is_hallucinated": "hallucination_rate"}, inplace=True)
    return hr

def compute_ps_mv(hr_df):
    """
    Compute Prompt Sensitivity (PS) and Model Variability (MV)
    PS = std deviation across prompt types for each model
    MV = std deviation across models for each prompt type
    """
    ps_scores = hr_df.groupby("model")["hallucination_rate"].std().reset_index()
    ps_scores.columns = ["model", "PS_score"]

    mv_scores = hr_df.groupby("prompt_type")["hallucination_rate"].std().reset_index()
    mv_scores.columns = ["prompt_type", "MV_score"]

    return ps_scores, mv_scores

# -------- SAVE METRICS -------- #
def save_metrics(hr_df, ps_df, mv_df):
    merged = hr_df.merge(ps_df, on="model")
    merged.to_csv(OUTPUT_METRICS_FILE, index=False)
    print(f"Saved metrics to {OUTPUT_METRICS_FILE}")
    return merged

# -------- VISUALIZATION -------- #
def plot_hr_bar(hr_df):
    pivot = hr_df.pivot(index="prompt_type", columns="model", values="hallucination_rate")
    pivot.plot(kind="bar", figsize=(10,6))
    plt.title("Hallucination Rate by Prompt Type")
    plt.ylabel("Hallucination Rate")
    plt.xlabel("Prompt Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hallucination_rate_by_prompt.png"))
    print("[✓] Saved: hallucination_rate_by_prompt.png")

def plot_ps_mv(ps_df, mv_df):
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    ax[0].bar(ps_df["model"], ps_df["PS_score"])
    ax[0].set_title("Prompt Sensitivity (PS)")
    ax[0].set_ylabel("Standard Deviation of HR")
    ax[0].set_xticklabels(ps_df["model"], rotation=45)

    ax[1].bar(mv_df["prompt_type"], mv_df["MV_score"])
    ax[1].set_title("Model Variability (MV)")
    ax[1].set_ylabel("Standard Deviation of HR")
    ax[1].set_xticklabels(mv_df["prompt_type"], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ps_mv_scores.png"))
    print("[✓] Saved: ps_mv_scores.png")

# -------- MAIN EXECUTION -------- #
def main():
    print("[•] Loading results...")
    df = load_all_results(RESULTS_DIR)

    if "hallucination_score" not in df.columns:
        raise ValueError("Missing 'hallucination_score' in input files. Please evaluate before attribution analysis.")

    print("[•] Computing hallucination rate...")
    hr_df = compute_hallucination_rate(df)

    print("[•] Computing PS and MV...")
    ps_df, mv_df = compute_ps_mv(hr_df)

    print("[•] Saving metrics...")
    merged_df = save_metrics(hr_df, ps_df, mv_df)

    print("[•] Generating plots...")
    plot_hr_bar(hr_df)
    plot_ps_mv(ps_df, mv_df)

    print("[✓] Attribution analysis complete.")

if __name__ == "__main__":
    main()
