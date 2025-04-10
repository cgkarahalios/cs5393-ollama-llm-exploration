import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def count_tokens(text):
    """Simple token counter using whitespace splitting - very approximate"""
    return len(text.split())


def analyze_response_times(results):
    """Analyze and visualize response times across models and categories"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())  # Get difficulty levels

    # Create a directory for charts if it doesn't exist
    charts_dir = "../results/charts"
    os.makedirs(charts_dir, exist_ok=True)

    # By category
    time_data = {model: [] for model in models}

    for model in models:
        for category in categories:
            # Now we have nested difficulty levels
            all_times = []
            for difficulty, prompts in results[model][category].items():
                for prompt_result in prompts:
                    if 'time_taken' in prompt_result:
                        all_times.append(prompt_result['time_taken'])

            time_data[model].append(np.mean(all_times) if all_times else 0)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(time_data, index=categories)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(kind='bar')
    plt.title('Average Response Time by Category and Model', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_times_by_category.png', dpi=300)
    plt.close()

    # By difficulty level (across all categories)
    time_data_by_difficulty = {model: [] for model in models}

    for model in models:
        for difficulty in difficulties:
            all_times = []
            for category in categories:
                for prompt_result in results[model][category][difficulty]:
                    if 'time_taken' in prompt_result:
                        all_times.append(prompt_result['time_taken'])

            time_data_by_difficulty[model].append(np.mean(all_times) if all_times else 0)

    # Create a DataFrame for easier plotting
    df_difficulty = pd.DataFrame(time_data_by_difficulty, index=difficulties)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = df_difficulty.plot(kind='bar')
    plt.title('Average Response Time by Difficulty and Model', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Difficulty', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fs', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_times_by_difficulty.png', dpi=300)
    plt.close()

    # Category and difficulty heatmap for each model
    for model in models:
        # Create a matrix of response times by category and difficulty
        heatmap_data = []
        for category in categories:
            category_data = []
            for difficulty in difficulties:
                times = [prompt_result['time_taken'] for prompt_result in results[model][category][difficulty]
                         if 'time_taken' in prompt_result]
                category_data.append(np.mean(times) if times else 0)
            heatmap_data.append(category_data)

        # Create a DataFrame for the heatmap
        df_heatmap = pd.DataFrame(heatmap_data, index=categories, columns=difficulties)

        # Plot
        plt.figure(figsize=(10, 8))
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#c6f7e9', '#2a9d8f'])
        im = plt.imshow(df_heatmap, cmap=cmap)
        plt.colorbar(im, label='Average Response Time (seconds)')
        plt.xticks(np.arange(len(difficulties)), difficulties, rotation=45)
        plt.yticks(np.arange(len(categories)), categories)
        plt.title(f'{model} - Response Time by Category and Difficulty', fontsize=14)

        # Add text annotations in the heatmap cells
        for i in range(len(categories)):
            for j in range(len(difficulties)):
                plt.text(j, i, f"{df_heatmap.iloc[i, j]:.1f}s",
                         ha="center", va="center", color="black", fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{charts_dir}/{model}_response_time_heatmap.png', dpi=300)
        plt.close()

    return df


def analyze_response_length(results):
    """Analyze and visualize response lengths (as token counts) across models and categories"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())

    # Create a directory for charts if it doesn't exist
    charts_dir = "../results/charts"
    os.makedirs(charts_dir, exist_ok=True)

    # By category
    length_data = {model: [] for model in models}

    for model in models:
        for category in categories:
            all_lengths = []
            for difficulty, prompts in results[model][category].items():
                for prompt_result in prompts:
                    if 'response' in prompt_result:
                        all_lengths.append(count_tokens(prompt_result['response']))

            length_data[model].append(np.mean(all_lengths) if all_lengths else 0)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(length_data, index=categories)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(kind='bar')
    plt.title('Average Response Length by Category and Model', fontsize=14)
    plt.ylabel('Approximate Token Count', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_lengths_by_category.png', dpi=300)
    plt.close()

    # By difficulty level (across all categories)
    length_data_by_difficulty = {model: [] for model in models}

    for model in models:
        for difficulty in difficulties:
            all_lengths = []
            for category in categories:
                for prompt_result in results[model][category][difficulty]:
                    if 'response' in prompt_result:
                        all_lengths.append(count_tokens(prompt_result['response']))

            length_data_by_difficulty[model].append(np.mean(all_lengths) if all_lengths else 0)

    # Create a DataFrame for easier plotting
    df_difficulty = pd.DataFrame(length_data_by_difficulty, index=difficulties)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = df_difficulty.plot(kind='bar')
    plt.title('Average Response Length by Difficulty and Model', fontsize=14)
    plt.ylabel('Approximate Token Count', fontsize=12)
    plt.xlabel('Difficulty', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/response_lengths_by_difficulty.png', dpi=300)
    plt.close()

    # Scatter plot of response time vs response length for each model
    for model in models:
        times = []
        lengths = []
        categories_list = []

        for category in categories:
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'response' in prompt_result and 'time_taken' in prompt_result:
                        times.append(prompt_result['time_taken'])
                        lengths.append(count_tokens(prompt_result['response']))
                        categories_list.append(category)

        # Create a DataFrame for the scatter plot
        df_scatter = pd.DataFrame({
            'Response Time': times,
            'Response Length': lengths,
            'Category': categories_list
        })

        # Plot
        plt.figure(figsize=(12, 8))

        # Different color for each category
        categories_set = set(categories_list)
        for category in categories_set:
            category_data = df_scatter[df_scatter['Category'] == category]
            plt.scatter(category_data['Response Length'], category_data['Response Time'],
                        label=category, alpha=0.7)

        # Add trend line
        if len(times) > 1:  # Need at least 2 points for regression
            z = np.polyfit(lengths, times, 1)
            p = np.poly1d(z)
            plt.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8,
                     label=f"Trend line: y={z[0]:.5f}x+{z[1]:.2f}")

        plt.title(f'{model} - Response Time vs Response Length', fontsize=14)
        plt.xlabel('Response Length (tokens)', fontsize=12)
        plt.ylabel('Response Time (seconds)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{charts_dir}/{model}_time_vs_length.png', dpi=300)
        plt.close()

    return df


def create_stats_tables(results):
    """Generate simple statistics tables as CSV files"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())

    stats_dir = "../results/stats"
    os.makedirs(stats_dir, exist_ok=True)

    # Response time statistics by model, category, and difficulty
    time_data = []
    for model in models:
        for category in categories:
            for difficulty in difficulties:
                difficulty_times = []
                for prompt_result in results[model][category][difficulty]:
                    if 'time_taken' in prompt_result:
                        difficulty_times.append(prompt_result['time_taken'])

                if difficulty_times:
                    time_data.append({
                        "Model": model,
                        "Category": category,
                        "Difficulty": difficulty,
                        "Avg Time (s)": np.mean(difficulty_times),
                        "Min Time (s)": np.min(difficulty_times),
                        "Max Time (s)": np.max(difficulty_times),
                        "Std Dev": np.std(difficulty_times)
                    })

    time_df = pd.DataFrame(time_data)
    time_df.to_csv(f"{stats_dir}/response_time_stats.csv", index=False)

    # Response length statistics
    length_data = []
    for model in models:
        for category in categories:
            for difficulty in difficulties:
                difficulty_lengths = []
                for prompt_result in results[model][category][difficulty]:
                    if 'response' in prompt_result:
                        difficulty_lengths.append(count_tokens(prompt_result['response']))

                if difficulty_lengths:
                    length_data.append({
                        "Model": model,
                        "Category": category,
                        "Difficulty": difficulty,
                        "Avg Length (tokens)": np.mean(difficulty_lengths),
                        "Min Length": np.min(difficulty_lengths),
                        "Max Length": np.max(difficulty_lengths),
                        "Std Dev": np.std(difficulty_lengths)
                    })

    length_df = pd.DataFrame(length_data)
    length_df.to_csv(f"{stats_dir}/response_length_stats.csv", index=False)

    return time_df, length_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_json_file>")
        sys.exit(1)

    results_file = sys.argv[1]

    # Make sure the results file exists
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found.")
        sys.exit(1)

    # Make sure the directories exist
    charts_dir = "../results/charts"
    os.makedirs(charts_dir, exist_ok=True)

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
    print(f"Results saved to the '../results/charts' directory")
    print(f"Statistics saved to the '../results/stats' directory")


if __name__ == "__main__":
    main()