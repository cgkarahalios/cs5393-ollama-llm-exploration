import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.colors import LinearSegmentedColormap


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
    charts_dir = "results/charts"
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


def analyze_memory_usage(results):
    """Analyze and visualize memory usage across models and categories"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())

    # Create a directory for charts if it doesn't exist
    charts_dir = "results/charts"
    os.makedirs(charts_dir, exist_ok=True)

    # By category
    memory_data = {model: [] for model in models}

    for model in models:
        for category in categories:
            all_memory = []
            for difficulty, prompts in results[model][category].items():
                for prompt_result in prompts:
                    if 'memory_usage' in prompt_result:
                        all_memory.append(prompt_result['memory_usage'])

            memory_data[model].append(np.mean(all_memory) if all_memory else 0)

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(memory_data, index=categories)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = df.plot(kind='bar')
    plt.title('Average Memory Usage by Category and Model', fontsize=14)
    plt.ylabel('Memory (MB)', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f MB', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/memory_usage_by_category.png', dpi=300)
    plt.close()

    # Model comparison bar chart for overall memory usage
    overall_memory = {model: [] for model in models}

    for model in models:
        all_memory = []
        for category in categories:
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'memory_usage' in prompt_result:
                        all_memory.append(prompt_result['memory_usage'])

        overall_memory[model] = np.mean(all_memory) if all_memory else 0

    # Create a DataFrame for the overall memory usage
    df_overall = pd.DataFrame(list(overall_memory.values()), index=list(overall_memory.keys()),
                              columns=['Memory Usage'])

    # Plot
    plt.figure(figsize=(10, 6))
    ax = df_overall.plot(kind='bar', color='#2a9d8f')
    plt.title('Overall Average Memory Usage by Model', fontsize=14)
    plt.ylabel('Memory (MB)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    ax.bar_label(ax.containers[0], fmt='%.1f MB', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{charts_dir}/overall_memory_usage.png', dpi=300)
    plt.close()

    return df


def analyze_response_length(results):
    """Analyze and visualize response lengths (as token counts) across models and categories"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())

    # Create a directory for charts if it doesn't exist
    charts_dir = "results/charts"
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


def generate_summary_report(results):
    """Generate a summary report of the results"""
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    difficulties = list(next(iter(results[models[0]].values())).keys())

    report = []

    # Add header
    report.append("# Ollama Model Evaluation Summary\n")
    report.append("*This report was automatically generated from model evaluation results.*\n\n")

    # Overall statistics
    report.append("## 1. Overall Statistics\n")

    # Response time table - by model and category
    time_data = []
    for model in models:
        model_times = []
        for category in categories:
            category_times = []
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'time_taken' in prompt_result:
                        category_times.append(prompt_result['time_taken'])
            model_times.append(np.mean(category_times) if category_times else 0)
        time_data.append([model] + [f"{t:.2f}s" for t in model_times] + [f"{np.mean(model_times):.2f}s"])

    report.append("### 1.1 Average Response Times by Category\n")
    report.append(tabulate(time_data, headers=['Model'] + categories + ['Overall'], tablefmt='pipe'))
    report.append("\n\n")

    # Response time table - by model and difficulty
    time_data_difficulty = []
    for model in models:
        model_times = []
        for difficulty in difficulties:
            difficulty_times = []
            for category in categories:
                for prompt_result in results[model][category][difficulty]:
                    if 'time_taken' in prompt_result:
                        difficulty_times.append(prompt_result['time_taken'])
            model_times.append(np.mean(difficulty_times) if difficulty_times else 0)
        time_data_difficulty.append([model] + [f"{t:.2f}s" for t in model_times] + [f"{np.mean(model_times):.2f}s"])

    report.append("### 1.2 Average Response Times by Difficulty\n")
    report.append(tabulate(time_data_difficulty, headers=['Model'] + difficulties + ['Overall'], tablefmt='pipe'))
    report.append("\n\n")

    # Memory usage table
    memory_data = []
    for model in models:
        model_memory = []
        for category in categories:
            category_memory = []
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'memory_usage' in prompt_result:
                        category_memory.append(prompt_result['memory_usage'])
            model_memory.append(np.mean(category_memory) if category_memory else 0)
        memory_data.append([model] + [f"{m:.2f}MB" for m in model_memory] + [f"{np.mean(model_memory):.2f}MB"])

    report.append("### 1.3 Average Memory Usage by Category\n")
    report.append(tabulate(memory_data, headers=['Model'] + categories + ['Overall'], tablefmt='pipe'))
    report.append("\n\n")

    # Response length table
    length_data = []
    for model in models:
        model_lengths = []
        for category in categories:
            category_lengths = []
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'response' in prompt_result:
                        category_lengths.append(count_tokens(prompt_result['response']))
            model_lengths.append(np.mean(category_lengths) if category_lengths else 0)
        length_data.append([model] + [f"{int(l)}" for l in model_lengths] + [f"{int(np.mean(model_lengths))}"])

    report.append("### 1.4 Average Response Lengths (approximate tokens) by Category\n")
    report.append(tabulate(length_data, headers=['Model'] + categories + ['Overall'], tablefmt='pipe'))
    report.append("\n\n")

    # Overall model performance
    overall_stats = []
    for model in models:
        all_times = []
        all_memory = []
        all_lengths = []

        for category in categories:
            for difficulty in difficulties:
                for prompt_result in results[model][category][difficulty]:
                    if 'time_taken' in prompt_result:
                        all_times.append(prompt_result['time_taken'])
                    if 'memory_usage' in prompt_result:
                        all_memory.append(prompt_result['memory_usage'])
                    if 'response' in prompt_result:
                        all_lengths.append(count_tokens(prompt_result['response']))

        avg_time = np.mean(all_times) if all_times else 0
        avg_memory = np.mean(all_memory) if all_memory else 0
        avg_length = np.mean(all_lengths) if all_lengths else 0

        overall_stats.append([
            model,
            f"{avg_time:.2f}s",
            f"{avg_memory:.2f}MB",
            f"{int(avg_length)}"
        ])

    report.append("### 1.5 Overall Model Performance\n")
    report.append(tabulate(overall_stats, headers=['Model', 'Avg Time', 'Avg Memory', 'Avg Tokens'], tablefmt='pipe'))
    report.append("\n\n")

    # Detailed results by category and difficulty
    report.append("## 2. Detailed Results by Category and Difficulty\n")

    for category in categories:
        report.append(f"### 2.{list(categories).index(category) + 1} {category.replace('_', ' ').title()}\n")

        for difficulty in difficulties:
            report.append(f"#### {difficulty.capitalize()} Difficulty\n")

            for model in models:
                report.append(f"##### {model}\n")

                for i, prompt_result in enumerate(results[model][category][difficulty]):
                    if 'response' in prompt_result:
                        # Truncate the prompt for display if it's too long
                        prompt_text = prompt_result['prompt']
                        if len(prompt_text) > 300:
                            prompt_text = prompt_text[:300] + "... [truncated]"

                        report.append(f"**Prompt {i + 1}**: {prompt_text}\n")
                        report.append(f"**Time**: {prompt_result['time_taken']:.2f}s\n")
                        report.append(f"**Memory**: {prompt_result['memory_usage']:.2f}MB\n")
                        report.append(f"**Tokens**: ~{count_tokens(prompt_result['response'])}\n")

                        # Limit response display to first 300 tokens to keep report manageable
                        response_text = prompt_result['response']
                        tokens = response_text.split()
                        if len(tokens) > 300:
                            response_text = ' '.join(tokens[:300]) + '... [truncated]'
                        report.append(f"**Response**:\n```\n{response_text}\n```\n\n")

    # Analysis section
    report.append("## 3. Analysis and Observations\n")
    report.append(
        "*This section should be filled in manually with your analysis. Consider the following questions:*\n\n")
    report.append("### 3.1 Response Quality\n")
    report.append("- Which model provided the highest quality responses for each category?\n")
    report.append("- How did response quality vary with difficulty level?\n")
    report.append("- Were there specific strengths or weaknesses observed for each model?\n\n")

    report.append("### 3.2 Performance Analysis\n")
    report.append("- How did model size correlate with response time and quality?\n")
    report.append("- Was there a notable difference in resource usage between models?\n")
    report.append("- Which model offered the best balance of quality vs. performance?\n\n")

    report.append("### 3.3 Task-Specific Observations\n")
    report.append("- Which tasks were most challenging for the models?\n")
    report.append("- Were there unexpected strengths or weaknesses for specific tasks?\n")
    report.append("- How did the models handle increasing complexity within each category?\n\n")

    # Write report to file
    report_dir = "results"
    os.makedirs(report_dir, exist_ok=True)
    with open(f'{report_dir}/evaluation_report.md', 'w') as f:
        f.write('\n'.join(report))

    print(f"Report generated: results/evaluation_report.md")


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
    charts_dir = "results/charts"
    os.makedirs(charts_dir, exist_ok=True)

    print(f"Loading results from {results_file}...")
    results = load_results(results_file)

    print("Analyzing response times...")
    time_df = analyze_response_times(results)
    print(time_df)

    print("\nAnalyzing memory usage...")
    memory_df = analyze_memory_usage(results)
    print(memory_df)

    print("\nAnalyzing response lengths...")
    length_df = analyze_response_length(results)
    print(length_df)

    print("\nGenerating summary report...")
    generate_summary_report(results)

    print("\nAnalysis complete!")
    print(f"Results saved to the 'results/charts' directory and 'results/evaluation_report.md'")


if __name__ == "__main__":
    main()