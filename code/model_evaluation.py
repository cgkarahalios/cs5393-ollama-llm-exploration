import json
import time
import os
import requests
from datetime import datetime
import nltk
from nltk.corpus import gutenberg, inaugural, reuters

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/gutenberg')
    nltk.data.find('corpora/inaugural')
    nltk.data.find('corpora/reuters')
except LookupError:
    nltk.download('gutenberg')
    nltk.download('inaugural')
    nltk.download('reuters')

# Models to test
MODELS = ["tinyllama", "mistral", "llama3.1"]

# Prepare text samples for summarization from NLTK corpus
# First join tokens into sentences, then join sentences
EASY_TEXT = ' '.join([' '.join(sent) for sent in inaugural.sents('2009-Obama.txt')[:5]])[:500]
MEDIUM_TEXT = ' '.join([' '.join(sent) for sent in reuters.sents(categories='trade')[:50]])[:1500]
HARD_TEXT = ' '.join([' '.join(sent) for sent in gutenberg.sents('melville-moby_dick.txt')[100:300]])[:3000]

# Test categories and prompts with difficulty levels
TEST_PROMPTS = {
    "general_qa": {
        "easy": [
            "What is the capital of France?"
        ],
        "medium": [
            "Explain how vaccines work to protect against diseases."
        ],
        "hard": [
            "If I buy a SPY call option and it ends up in the money at expiration, but "
            "I don't have enough cash to "
            "actually purchase the shares, what happens? Can I still "
            "profit from the trade without exercising the option?"
        ]
    },
    "summarization": {
        "easy": [
            f"Summarize this text in a few sentences: {EASY_TEXT}"
        ],
        "medium": [
            f"Provide a concise summary of the following text, capturing the main points: {MEDIUM_TEXT}"
        ],
        "hard": [
            f"Create a detailed summary of this complex text, organizing the main themes and arguments: {HARD_TEXT}"
        ]
    },
    "code_generation": {
        "easy": [
            "Write a Python function to calculate the factorial of a number."
        ],
        "medium": [
            "Write a Python function that implements a binary search algorithm for a sorted list."
        ],
        "hard": [
            "Create the most efficient C++ program to calculate the nth Fibonacci number using matrix exponentiation with a time complexity better than O(n)."
        ]
    },
    "creative_writing": {
        "easy": [
            "Write a short story about a dog who finds a treasure."
        ],
        "medium": [
            "Write a short story about modern historical events with references and allusions towards greek history and famous greek figures/myths"
        ],
        "hard": [
            "Write a short story that uses non-linear narrative structure to explore the theme of memory. Include at least three different time periods and ensure they interconnect meaningfully."
        ]
    }
}


def run_ollama(model, prompt):
    """Run Ollama with the given model and prompt using HTTP API - NO TIMEOUT, only tracking time"""
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    print(f"      Starting request to {model}...")
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


def main():
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories for output if they don't exist
    os.makedirs("../results/raw", exist_ok=True)

    print("Starting model evaluation...")
    print(f"Models to test: {', '.join(MODELS)}")
    print(
        f"Number of prompts: {sum(len(prompts) for category in TEST_PROMPTS.values() for difficulty, prompts in category.items())}")

    for model in MODELS:
        print(f"\n{'=' * 50}")
        print(f"Testing model: {model}")
        print(f"{'=' * 50}")

        model_results = {}

        for category, difficulty_levels in TEST_PROMPTS.items():
            print(f"\nCategory: {category}")
            category_results = {}

            for difficulty, prompts in difficulty_levels.items():
                print(f"  Difficulty: {difficulty}")
                difficulty_results = []

                for i, prompt in enumerate(prompts):
                    print(f"    Running prompt {i + 1}/{len(prompts)}...")
                    prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    print(f"    Prompt: {prompt_display}")

                    try:
                        result = run_ollama(model, prompt)

                        if "error" in result and result["error"]:
                            print(f"      Error: {result['error']}")
                        else:
                            response_length = len(result["output"].split())
                            print(f"      Completed in {result['time_taken']:.2f} seconds")
                            print(f"      Response length: {response_length} words")
                            # Print first 50 chars of response for quick verification
                            first_part = result["output"][:50].replace('\n', ' ')
                            print(f"      Preview: {first_part}...")

                        difficulty_results.append({
                            "prompt": prompt,
                            "response": result.get("output", ""),
                            "time_taken": result["time_taken"]
                        })
                    except Exception as e:
                        print(f"      Error: {str(e)}")
                        difficulty_results.append({
                            "prompt": prompt,
                            "error": str(e)
                        })

                category_results[difficulty] = difficulty_results

            model_results[category] = category_results

        results[model] = model_results

    # Save results to a JSON file
    output_filename = f"../results/raw/ollama_results_{timestamp}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete!")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()