
# Hallucination Attribution in Large Language Models (LLMs)

This repository provides a fully open-source, reproducible pipeline to investigate hallucinations in LLMs and attribute their cause to prompt design or model behavior.

## 📂 Directory Structure
- `data/`: Sample benchmark datasets.
- `prompts/`: Prompt templates for each variation type.
- `scripts/`: Python scripts for generation, evaluation, and analysis.
- `results/`: Output of generations and evaluation.
- `notebooks/`: Optional Colab/Kaggle-ready notebooks.
- `models/`: Load configurations for LLaMA 2, Mistral, DeepSeek, etc.

---

## 📋 Requirements

- Python >= 3.8
- pip install -r requirements.txt

## 🔧 Installation

```bash
git clone https://github.com/yourusername/llm_hallucination_experiment.git
cd llm_hallucination_experiment
pip install -r requirements.txt
```

---

## 🚀 How to Run Experiments

### Step 1: Load Dataset
Edit `scripts/load_dataset.py` to select dataset: TruthfulQA, QAFactEval, HallucinationEval.

### Step 2: Choose Prompt Variant
Templates in `prompts/`:
- zero_shot.txt
- few_shot.txt
- instruction.txt
- cot.txt
- vague.txt

### Step 3: Generate Outputs
Run generation script:
```bash
python scripts/generate.py --model mistral --dataset truthfulqa --prompt few_shot
```

### Step 4: Evaluate Hallucinations
```bash
python scripts/evaluate.py --input results/mistral_truthfulqa_fewshot.json
```

### Step 5: Attribution Analysis
```bash
python scripts/attribution_analysis.py
```

---

## 📈 Metrics Computed
- Hallucination Rate (HR)
- Prompt Sensitivity (PS)
- Model Variability (MV)

## 📊 Visualization
Output graphs include:
- HR by prompt type (bar chart)
- Radar plots of model comparison
- PS vs MV Attribution quadrant

---

## 📚 Data Sources
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa)
- [QAFactEval](https://github.com/salesforce/QAFactEval)
- [HallucinationEval](https://huggingface.co/datasets/hallucination_eval)
- [CohS](https://huggingface.co/datasets/cais/multi-fact-summ-cohs)

---

## 🧠 Models Used
All models loaded from HuggingFace:
- `meta-llama/Llama-2-13b-chat-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `deepseek-ai/deepseek-llm-67b-chat`
- `openchat/openchat-3.5`

---

## 📜 License
MIT License - Free to use and modify.

---
