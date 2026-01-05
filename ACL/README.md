
# FinVQA-Chart Evaluation Suite

This repository contains the implementation of the evaluation harness for the **FinVQA-Chart** dataset (Financial Vision-Language Reasoning at Scale).

## Dataset
The dataset is sourced from Hugging Face: [qsWDFGHN/FIN-VQA](https://huggingface.co/datasets/qsWDFGHN/FIN-VQA).
It focuses on evaluating LVLMs on financial chart understanding tasks using metrics like:
- Overall Accuracy
- Accuracy by Question Type (Pattern Recognition, Volume Analysis, etc.)
- Accuracy by Difficulty (Easy, Medium, Hard)

## Models Implemented
The following models commonly evaluated in this benchmark are implemented:

- **Proprietary Models**:
    - `GPT-4o` (OpenAI)
    - `GPT-4V` (OpenAI)
    - `Claude 3 Opus` (Anthropic)
    - `Claude 3.5 Sonnet` (Anthropic)
    - `Gemini 1.5 Pro` (Google)
    - `Gemini 1.5 Flash` (Google)
    
- **Open Source Models**:
    - `Qwen2-VL-72B`
    - `InternVL2-26B`
    - `LLaVA-1.6-34B`
    - `CogVLM2-19B`
    - `DeepSeek-VL-7B`
    - `InternLM-XComposer2`
    - `GLM-4V-9B`
    - `MiniCPM-V-2.6`
    - `Phi-3.5-Vision`
    - `DePlot` (Chart-to-Table baseline)

## structure
- `dataset.py`: Handles loading the FinVQA dataset from Hugging Face.
- `eval.py`: Main evaluation script computing metrics.
- `models/`: Directory containing model wrapper implementations.

## Usage

1. Install dependencies:
```bash
pip install datasets openai google-generativeai anthropic transformers torch torchvision pdb
```

2. Run evaluation:
```bash
python eval.py --model gpt4o
python eval.py --model claude_opus
python eval.py --model qwen2_vl
python eval.py --model phi3_vision
```

## Note
For proprietary models, ensure you have set the corresponding environment variables:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
