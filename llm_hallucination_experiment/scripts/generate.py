"""
generate.py

Loads a selected open-source LLM and prompt template,
applies it to each sample in the dataset,
and generates model responses for attribution analysis.
"""

import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# -------- CONFIG -------- #
PROMPT_DIR = "prompts/"
DATA_DIR = "data/"
RESULT_DIR = "results/"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------- UTILS -------- #

def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_prompt_template(prompt_file):
    with open(prompt_file, "r") as f:
        return f.read()

def format_prompt(template, question):
    return template.replace("{question}", question)

def load_model(model_name):
    print(f"[•] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

# -------- GENERATE -------- #

def generate_responses(generator, data, prompt_template, model_name, prompt_type, max_tokens=256):
    results = []
    for item in tqdm(data, desc="Generating responses"):
        formatted_prompt = format_prompt(prompt_template, item["question"])
        response = generator(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]['generated_text']

        results.append({
            "model": model_name,
            "prompt_type": prompt_type,
            "question": item["question"],
            "formatted_prompt": formatted_prompt,
            "response": response.strip()
        })
    return results

# -------- MAIN -------- #

def main():
    parser = argparse.ArgumentParser(description="Generate responses from LLMs for attribution experiment")
    parser.add_argument("--model", required=True, help="HuggingFace model ID (e.g., mistralai/Mistral-7B-Instruct-v0.1)")
    parser.add_argument("--dataset", required=True, help="Path to preprocessed JSON dataset file")
    parser.add_argument("--prompt_type", required=True, help="Prompt type name (matches .txt file in prompts/)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--output", help="Custom output file name (optional)")

    args = parser.parse_args()

    dataset_path = os.path.join(DATA_DIR, args.dataset)
    prompt_path = os.path.join(PROMPT_DIR, f"{args.prompt_type}.txt")

    data = load_dataset(dataset_path)
    prompt_template = load_prompt_template(prompt_path)
    generator = load_model(args.model)

    print("[•] Generating...")
    results = generate_responses(generator, data, prompt_template, args.model, args.prompt_type, max_tokens=args.max_tokens)

    output_path = args.output or f"{RESULT_DIR}/{args.model.replace('/', '_')}_{args.prompt_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Saved results to {output_path}")

if __name__ == "__main__":
    main()
