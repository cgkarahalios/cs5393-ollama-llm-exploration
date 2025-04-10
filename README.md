# Ollama LLM Exploration

This repository contains code and results from an exploration of open-source Large Language Models (LLMs) using Ollama.

## Important Documentation

### Written Report
For a comprehensive analysis of the findings, please refer to:
- `report/Ollama_LLM_Exploration_Report.pdf` - Contains the full written analysis, methodology, findings, and conclusions

### LLM Output Reports
To see the actual outputs from all LLMs in response to the various test questions:
- `results/reports/model_evaluation_report.pdf` - Contains all model responses to each prompt across categories
- `results/reports/prompt_engineering_report.pdf` - Shows detailed outputs from different prompting techniques

These PDF reports provide the complete responses from each model, allowing you to directly compare output quality, accuracy, and style.

## Models Evaluated

The following models were evaluated:
- **TinyLlama** (1.1B parameters)
- **Mistral** (7B parameters)
- **Llama3.1** (8B parameters)

## Repository Structure

### Code

- `code/model_evaluation.py` - **Main script** for evaluating model performance across different tasks and difficulties
- `code/focused_experiments.py` - Script for prompt engineering experiments
- `code/analyze_results.py` - Script for analyzing model evaluation results and generating charts
- `code/analyze_prompt_experiments.py` - Script for analyzing prompt engineering results
- `code/create_report_pdf.py` - Script for generating PDF reports from experiment results

### Results

- `results/raw/` - Contains raw JSON output from model evaluations
- `results/stats/` - Contains CSV files with statistical analysis
- `results/charts/` - Contains visualization PNGs of model performance
- `results/prompt_engineering/` - Contains prompt engineering experiment results and charts

## Running the Code

1. **Download Models**
   ```bash
   ollama pull tinyllama
   ollama pull mistral
   ollama pull llama3.1
   ```

2. **Run Evaluations**
   ```bash
   cd code
   python model_evaluation.py
   ```

3. **Analyze Results**
   ```bash
   python analyze_results.py ../results/raw/ollama_results_[timestamp].json
   ```

## Key Findings

- **TinyLlama**: Fastest response times (avg 5.47s) but lower quality outputs
- **Mistral**: Good balance of speed (avg 19.85s) and quality
- **Llama3.1**: Highest quality outputs but slowest response times (avg 84.69s)

Creative writing and code generation were found to be the most time-intensive tasks, while summarization and general Q&A were typically faster across all models.

Few-shot prompting was found to be the most efficient technique across all models.
