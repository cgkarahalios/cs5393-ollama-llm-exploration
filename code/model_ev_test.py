#!/usr/bin/env python3
"""
Simplified model evaluation script for testing a single model with fewer prompts
"""
import json
import time
import psutil
import subprocess
import os
from datetime import datetime

# Just test one model initially
MODEL = "tinyllama"  # Change to the model you want to test

# Simplified prompt set - just one prompt per category and difficulty
TEST_PROMPTS = {
    "general_qa": {
        "easy": ["What is photosynthesis?"],
        "medium": ["Explain how vaccines work to protect against diseases."],
        "hard": [
            "Explain the double-slit experiment in quantum physics and its implications for our understanding of light."]
    },
    "code_generation": {
        "easy": ["Write a Python function to calculate the factorial of a number."],
        "medium": ["Write a Python function that implements a binary search algorithm for a sorted list."],
        "hard": ["Create the most efficient C++ program to calculate the nth Fibonacci number."]
    },
    "creative_writing": {
        "easy": ["Write a short story about a dog who finds a treasure."],
        "medium": [
            "Write a short story that begins with: 'The door creaked open, revealing a room that hadn't been entered in decades.'"],
        "hard": ["Write a short story that uses non-linear narrative structure to explore the theme of memory."]
    }
}


import requests

def run_ollama(model, prompt, timeout=60):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False  # easier to handle for evaluation
    }

    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        end_time = time.time()

        if response.status_code == 200:
            result_json = response.json()
            return {
                "output": result_json.get("response", ""),
                "error": "",
                "time_taken": end_time - start_time,
                "memory_usage": 0,  # You can still track this with psutil if needed
                "cpu_usage": 0
            }
        else:
            return {
                "output": "",
                "error": f"HTTP {response.status_code}: {response.text}",
                "time_taken": end_time - start_time,
                "memory_usage": 0,
                "cpu_usage": 0
            }
    except requests.exceptions.Timeout:
        return {
            "output": "[Timeout: Model took too long to respond]",
            "error": "Request timed out",
            "time_taken": timeout,
            "memory_usage": 0,
            "cpu_usage": 0
        }

def main():
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories for output if they don't exist
    os.makedirs("results/raw", exist_ok=True)

    print(f"Testing model: {MODEL}")

    model_results = {}

    for category, difficulty_levels in TEST_PROMPTS.items():
        print(f"\nCategory: {category}")
        category_results = {}

        for difficulty, prompts in difficulty_levels.items():
            print(f"  Difficulty: {difficulty}")
            difficulty_results = []

            for i, prompt in enumerate(prompts):
                print(f"    Running prompt {i + 1}/{len(prompts)}...")
                print(f"    Prompt: {prompt}")

                try:
                    result = run_ollama(MODEL, prompt)
                    if result["output"] == "[Timeout: Model took too long to respond]":
                        print(f"      ⚠️ TIMEOUT after {result['time_taken']} seconds")
                    else:
                        response_length = len(result["output"].split())
                        print(f"      Completed in {result['time_taken']:.2f} seconds")
                        print(f"      Response length: {response_length} words")

                    difficulty_results.append({
                        "prompt": prompt,
                        "response": result["output"],
                        "time_taken": result["time_taken"],
                        "memory_usage": result["memory_usage"],
                        "cpu_usage": result["cpu_usage"]
                    })
                except Exception as e:
                    print(f"      Error: {str(e)}")
                    difficulty_results.append({
                        "prompt": prompt,
                        "error": str(e)
                    })

            category_results[difficulty] = difficulty_results

        model_results[category] = category_results

    results[MODEL] = model_results

    # Save results to a JSON file
    output_filename = f"results/raw/ollama_simple_test_{timestamp}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()