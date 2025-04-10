#!/usr/bin/env python3
"""
Focused experiment script for testing prompt engineering techniques across models
WITHOUT ANY TIMEOUT - will run until completion, tracking only time
"""
import json
import time
import os
import requests
from datetime import datetime

# Models to test
MODELS = ["tinyllama", "mistral", "llama3.1"]

# Different prompting techniques to evaluate
PROMPTING_TECHNIQUES = {
    "basic": {
        "description": "Standard direct questioning",
        "template": "{question}"
    },
    "few_shot": {
        "description": "Providing examples before asking the question",
        "template": """Here are some examples:

Question: What is the capital of Spain?
Answer: The capital of Spain is Madrid.

Question: What is the capital of Italy?
Answer: The capital of Italy is Rome.

Question: What is the capital of Germany?
Answer: The capital of Germany is Berlin.

Now, please answer the following question:
Question: {question}
Answer:"""
    },
    "chain_of_thought": {
        "description": "Encouraging step-by-step reasoning",
        "template": """Please think through this step by step:

{question}

Let's work through this logically:"""
    },
    "role_based": {
        "description": "Assigning a specific role to the model",
        "template": """You are an expert in {domain} with many years of experience.

A student has asked you the following question:
{question}

Please provide your expert answer:"""
    },
    "self_consistency": {
        "description": "Asking for multiple approaches/perspectives",
        "template": """Consider the following question from multiple perspectives:

{question}

Approach 1:
Approach 2:
Approach 3:

Based on these approaches, the most consistent answer is:"""
    }
}

# Test questions with appropriate domains for role-based prompting
TEST_QUESTIONS = [
    {
        "question": "What are the implications of rising global temperatures?",
        "domain": "climate science",
        "category": "general_qa"
    },
    {
        "question": "How do you determine the time complexity of a recursive algorithm?",
        "domain": "computer science",
        "category": "general_qa"
    },
    {
        "question": "Write a Python function to find the most frequent element in a list.",
        "domain": "programming",
        "category": "code_generation"
    },
    {
        "question": "Explain how blockchain technology works and its potential applications beyond cryptocurrency.",
        "domain": "blockchain technology",
        "category": "general_qa"
    }
]


def run_ollama(model, prompt):
    """Run Ollama with the given model and prompt using HTTP API - NO TIMEOUT, only tracking time"""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    print(f"    Starting request to {model}...")
    start_time = time.time()

    try:
        # No timeout parameter - will run until completion
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()

        if response.status_code == 200:
            result_json = response.json()
            return {
                "output": result_json.get("response", ""),
                "time_taken": end_time - start_time
            }
        else:
            return {
                "output": "",
                "error": f"HTTP {response.status_code}: {response.text}",
                "time_taken": end_time - start_time
            }
    except Exception as e:
        end_time = time.time()
        return {
            "output": "",
            "error": str(e),
            "time_taken": end_time - start_time
        }


def format_prompt(question_data, technique):
    """Format the prompt according to the specified technique"""
    template = PROMPTING_TECHNIQUES[technique]["template"]

    # Replace placeholders in the template
    prompt = template.format(
        question=question_data["question"],
        domain=question_data.get("domain", "general knowledge")
    )

    return prompt


def main():
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories for output if they don't exist
    os.makedirs("../results/prompt_engineering", exist_ok=True)

    print("Starting prompt engineering experiments...")
    print(f"Models to test: {', '.join(MODELS)}")
    print(f"Prompting techniques: {', '.join(PROMPTING_TECHNIQUES.keys())}")
    print(f"Number of questions: {len(TEST_QUESTIONS)}")

    for model in MODELS:
        print(f"\n{'=' * 50}")
        print(f"Testing model: {model}")
        print(f"{'=' * 50}")

        model_results = {}

        for question_data in TEST_QUESTIONS:
            question = question_data["question"]
            category = question_data["category"]

            print(f"\nQuestion: {question}")
            question_results = {}

            for technique, technique_info in PROMPTING_TECHNIQUES.items():
                print(f"  Technique: {technique} - {technique_info['description']}")

                # Format the prompt according to the technique
                prompt = format_prompt(question_data, technique)

                try:
                    # Run the model with the formatted prompt - no timeout
                    result = run_ollama(model, prompt)

                    if "error" in result and result["error"]:
                        print(f"    Error: {result['error']}")
                    else:
                        response_length = len(result["output"].split())
                        print(f"    Completed in {result['time_taken']:.2f} seconds")
                        print(f"    Response length: {response_length} words")
                        # Print first 50 chars of response for quick verification
                        first_part = result["output"][:50].replace('\n', ' ')
                        print(f"    Preview: {first_part}...")

                    question_results[technique] = {
                        "prompt": prompt,
                        "response": result.get("output", ""),
                        "time_taken": result["time_taken"]
                    }
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    question_results[technique] = {
                        "prompt": prompt,
                        "error": str(e)
                    }

            model_results[question] = {
                "question_data": question_data,
                "results": question_results
            }

        results[model] = model_results

    # Save results to a JSON file
    output_filename = f"../results/prompt_engineering/prompt_techniques_{timestamp}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPrompt engineering experiments complete!")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()