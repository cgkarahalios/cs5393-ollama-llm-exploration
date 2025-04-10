"""
Fixed analysis script for prompt engineering experiment results - plots and stats only
"""
import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def count_tokens(text):
    """Simple token counter using NLTK tokenization"""
    return len(word_tokenize(text)) if text else 0

def analyze_response_times(results):
    """Analyze and visualize response times across models and techniques"""
    models = list(results.keys())

    # Get all unique questions and techniques
    questions = []
    techniques = set()
    for model in models:
        questions.extend(list(results[model].keys()))
        for question in results[model]:
            techniques.update(results[model][question]["results"].keys())

    questions = list(set(questions))
    techniques = list(techniques)

    # Create a directory for charts if it doesn't exist
    charts_dir = "../results/prompt_engineering/charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Prepare data for plotting
    technique_times = {model: {technique: [] for technique in techniques} for model in models}

    for model in models:
        for question in results[model]:
            for technique in techniques:
                if technique in results[model][question]["results"]:
                    result = results[model][question]["results"][technique]
                    if "time_taken" in result:
                        technique_times[model][technique].append(result["time_taken"])

    # Calculate average times
    avg_times = {model: {} for model in models}
    for model in models:
        for technique in techniques:
            times = technique_times[model][technique]
            avg_times[model][technique] = np.mean(times) if times else 0

    # Create a DataFrame for plotting
    data = []
    for model in models:
        for technique in techniques:
            data.append({
                "Model": model,
                "Technique": technique,
                "Average Response Time (s)": avg_times[model][technique]
            })

    df = pd.DataFrame(data)

    # Plot average response times by technique and model
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Technique", y="Average Response Time (s)", hue="Model", data=df)
    plt.title('Average Response Time by Prompting Technique', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Technique', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_times_by_technique.png', dpi=300)
    plt.close()

    # Create a heatmap of response times
    pivot_df = df.pivot(index="Technique", columns="Model", values="Average Response Time (s)")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Response Time Heatmap by Model and Technique', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_time_heatmap.png', dpi=300)
    plt.close()

    return df

def analyze_response_length(results):
    """Analyze and visualize response lengths across models and techniques"""
    models = list(results.keys())

    # Get all unique questions and techniques
    questions = []
    techniques = set()
    for model in models:
        questions.extend(list(results[model].keys()))
        for question in results[model]:
            techniques.update(results[model][question]["results"].keys())

    questions = list(set(questions))
    techniques = list(techniques)

    # Create a directory for charts if it doesn't exist
    charts_dir = "../results/prompt_engineering/charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Prepare data for plotting
    technique_lengths = {model: {technique: [] for technique in techniques} for model in models}

    for model in models:
        for question in results[model]:
            for technique in techniques:
                if technique in results[model][question]["results"]:
                    result = results[model][question]["results"][technique]
                    if "response" in result:
                        technique_lengths[model][technique].append(len(result["response"].split()))

    # Calculate average lengths
    avg_lengths = {model: {} for model in models}
    for model in models:
        for technique in techniques:
            lengths = technique_lengths[model][technique]
            avg_lengths[model][technique] = np.mean(lengths) if lengths else 0

    # Create a DataFrame for plotting
    data = []
    for model in models:
        for technique in techniques:
            data.append({
                "Model": model,
                "Technique": technique,
                "Average Response Length (words)": avg_lengths[model][technique]
            })

    df = pd.DataFrame(data)

    # Plot average response lengths by technique and model
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Technique", y="Average Response Length (words)", hue="Model", data=df)
    plt.title('Average Response Length by Prompting Technique', fontsize=14)
    plt.ylabel('Words', fontsize=12)
    plt.xlabel('Technique', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Model")

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_lengths_by_technique.png', dpi=300)
    plt.close()

    # Scatter plot of response time vs length for each model
    for model in models:
        times = []
        lengths = []
        techniques_list = []

        for question in results[model]:
            for technique in techniques:
                if technique in results[model][question]["results"]:
                    result = results[model][question]["results"][technique]
                    if "response" in result and "time_taken" in result:
                        times.append(result["time_taken"])
                        lengths.append(len(result["response"].split()))
                        techniques_list.append(technique)

        scatter_df = pd.DataFrame({
            "Response Time (s)": times,
            "Response Length (words)": lengths,
            "Technique": techniques_list
        })

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x="Response Length (words)", y="Response Time (s)",
                        hue="Technique", style="Technique", data=scatter_df)

        # Add trend line
        if len(times) > 1:
            z = np.polyfit(lengths, times, 1)
            p = np.poly1d(z)
            plt.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8,
                      label=f"Trend line: y={z[0]:.5f}x+{z[1]:.2f}")

        plt.title(f'{model} - Response Time vs Response Length', fontsize=14)
        plt.xlabel('Response Length (words)', fontsize=12)
        plt.ylabel('Response Time (seconds)', fontsize=12)
        plt.legend(title="Technique")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{charts_dir}/{model}_time_vs_length.png', dpi=300)
        plt.close()

    return df

def create_stats_tables(results):
    """Generate simple statistics tables as CSV files"""
    models = list(results.keys())

    # Get all unique questions and techniques
    questions = []
    techniques = set()
    for model in models:
        questions.extend(list(results[model].keys()))
        for question in results[model]:
            techniques.update(results[model][question]["results"].keys())

    questions = list(set(questions))
    techniques = list(techniques)

    stats_dir = "../results/prompt_engineering/stats"
    os.makedirs(stats_dir, exist_ok=True)

    # Response time statistics
    time_data = []
    for model in models:
        for technique in techniques:
            technique_times = []
            for question in results[model]:
                if technique in results[model][question]["results"]:
                    result = results[model][question]["results"][technique]
                    if "time_taken" in result:
                        technique_times.append(result["time_taken"])

            if technique_times:
                time_data.append({
                    "Model": model,
                    "Technique": technique,
                    "Avg Time (s)": np.mean(technique_times),
                    "Min Time (s)": np.min(technique_times),
                    "Max Time (s)": np.max(technique_times),
                    "Std Dev": np.std(technique_times)
                })

    time_df = pd.DataFrame(time_data)
    time_df.to_csv(f"{stats_dir}/response_time_stats.csv", index=False)

    # Response length statistics
    length_data = []
    for model in models:
        for technique in techniques:
            technique_lengths = []
            for question in results[model]:
                if technique in results[model][question]["results"]:
                    result = results[model][question]["results"][technique]
                    if "response" in result:
                        technique_lengths.append(len(result["response"].split()))

            if technique_lengths:
                length_data.append({
                    "Model": model,
                    "Technique": technique,
                    "Avg Length (words)": np.mean(technique_lengths),
                    "Min Length": np.min(technique_lengths),
                    "Max Length": np.max(technique_lengths),
                    "Std Dev": np.std(technique_lengths)
                })

    length_df = pd.DataFrame(length_data)
    length_df.to_csv(f"{stats_dir}/response_length_stats.csv", index=False)

    return time_df, length_df

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_prompt_experiments.py <results_json_file>")
        sys.exit(1)

    results_file = sys.argv[1]

    # Make sure the results file exists
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found.")
        sys.exit(1)

    # Make sure the directories exist
    charts_dir = "../results/prompt_engineering/charts"
    stats_dir = "../results/prompt_engineering/stats"
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    print(f"Loading results from {results_file}...")
    results = load_results(results_file)

    print("Analyzing response times...")
    time_df = analyze_response_times(results)
    print("Time analysis complete.")

    print("\nAnalyzing response lengths...")
    length_df = analyze_response_length(results)
    print("Length analysis complete.")

    print("\nGenerating statistics tables...")
    create_stats_tables(results)
    print("Statistics tables generated.")

    print("\nAnalysis complete!")
    print(f"Results saved to the '../results/prompt_engineering/charts' directory")
    print(f"Statistics saved to the '../results/prompt_engineering/stats' directory")


if __name__ == "__main__":
    main()