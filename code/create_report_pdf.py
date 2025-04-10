"""
Script to generate human-readable PDF reports from the experiment results
with proper Unicode handling
"""
import json
import sys
import os
import argparse
import re
from datetime import datetime
from fpdf import FPDF
import textwrap


# Function to sanitize text for FPDF (which uses latin-1 encoding)
def sanitize_text(text):
    if text is None:
        return ""

    # Replace common Unicode characters with ASCII equivalents
    replacements = {
        '\u2014': '-',  # em dash
        '\u2013': '-',  # en dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',  # non-breaking space
        # Add more replacements as needed
    }

    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Remove any remaining non-Latin1 characters
    return ''.join(c if ord(c) < 256 else '?' for c in text)


class PDF(FPDF):
    def __init__(self, title="Ollama Experiment Report"):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, sanitize_text(title), 0, 1, 'L')
        self.ln(5)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, sanitize_text(title), 0, 1, 'L')

    def sub_section_title(self, title):
        self.set_font('Arial', 'BI', 11)
        self.cell(0, 10, sanitize_text(title), 0, 1, 'L')

    def body_text(self, text):
        self.set_font('Arial', '', 10)

        # Sanitize text for PDF
        text = sanitize_text(text)

        # Split text into lines that fit within the page width
        wrapped_text = textwrap.fill(text, width=95)
        lines = wrapped_text.split('\n')

        for line in lines:
            self.multi_cell(0, 5, line)
        self.ln(5)

    def code_text(self, text):
        self.set_font('Courier', '', 9)

        # Sanitize text for PDF
        text = sanitize_text(text)

        # Split code into lines
        lines = text.split('\n')

        for line in lines:
            # Trim very long lines
            if len(line) > 100:
                line = line[:97] + "..."
            self.cell(0, 5, line, 0, 1)
        self.ln(5)


def load_results(filename):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def create_model_evaluation_report(results_file, output_file):
    """Create a PDF report for the model evaluation results"""
    # Load the results
    results = load_results(results_file)

    # Create PDF object
    pdf = PDF("Ollama Model Evaluation Report")
    pdf.set_author("Ollama Experiment")

    # Add timestamp
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.cell(0, 5, f"Data source: {os.path.basename(results_file)}", 0, 1, 'R')
    pdf.ln(10)

    # Add introduction
    pdf.chapter_title("Introduction")
    pdf.body_text("This report contains the results of evaluating different open-source LLMs using Ollama. "
                  "The models were tested on various tasks with different difficulty levels.")
    pdf.ln(5)

    # Add models info
    models = list(results.keys())
    pdf.chapter_title("Models Evaluated")
    pdf.body_text(f"The following models were evaluated: {', '.join(models)}")
    pdf.ln(10)

    # Report for each model
    for model in models:
        pdf.add_page()
        pdf.chapter_title(f"Model: {model}")

        categories = results[model].keys()

        for category in categories:
            pdf.section_title(f"Category: {category}")

            difficulties = results[model][category].keys()

            for difficulty in difficulties:
                pdf.sub_section_title(f"Difficulty: {difficulty}")

                for i, prompt_result in enumerate(results[model][category][difficulty]):
                    # Add page break if needed for new prompt
                    if pdf.get_y() > 220:
                        pdf.add_page()

                    pdf.set_font('Arial', 'B', 10)
                    prompt_num = i + 1
                    pdf.cell(0, 5, f"Prompt {prompt_num}:", 0, 1)

                    pdf.set_font('Arial', 'I', 10)

                    # Show prompt (truncate if too long)
                    prompt = prompt_result.get("prompt", "")
                    if len(prompt) > 300:
                        prompt = prompt[:297] + "..."
                    pdf.multi_cell(0, 5, sanitize_text(prompt))

                    # Response time
                    time_taken = prompt_result.get("time_taken", 0)
                    pdf.set_font('Arial', '', 10)
                    pdf.cell(0, 5, f"Response time: {time_taken:.2f} seconds", 0, 1)

                    # Response
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 5, "Response:", 0, 1)

                    response = prompt_result.get("response", "")

                    # Check if this is code
                    if category == "code_generation":
                        pdf.code_text(response)
                    else:
                        # For normal text responses
                        pdf.body_text(response)

                    pdf.ln(10)

    # Save the PDF
    pdf.output(output_file)
    print(f"Model evaluation report saved to: {output_file}")


def create_prompt_engineering_report(results_file, output_file):
    """Create a PDF report for the prompt engineering results"""
    # Load the results
    results = load_results(results_file)

    # Create PDF object
    pdf = PDF("Ollama Prompt Engineering Experiment Report")
    pdf.set_author("Ollama Experiment")

    # Add timestamp
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.cell(0, 5, f"Data source: {os.path.basename(results_file)}", 0, 1, 'R')
    pdf.ln(10)

    # Add introduction
    pdf.chapter_title("Introduction")
    pdf.body_text("This report contains the results of testing different prompt engineering techniques "
                  "with open-source LLMs using Ollama. Various techniques were compared to evaluate "
                  "their effectiveness across different models and question types.")
    pdf.ln(5)

    # Add models info
    models = list(results.keys())
    pdf.chapter_title("Models Evaluated")
    pdf.body_text(f"The following models were evaluated: {', '.join(models)}")
    pdf.ln(10)

    # Get all techniques and questions
    techniques = set()
    questions = set()
    for model in models:
        for question in results[model]:
            questions.add(question)
            for technique in results[model][question]["results"]:
                techniques.add(technique)

    pdf.chapter_title("Prompt Engineering Techniques")
    for technique in techniques:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 5, f"{technique.title()}", 0, 1)

        # Add technique description if we had one
        if technique == "basic":
            pdf.body_text("Standard direct questioning without any specific prompt engineering.")
        elif technique == "few_shot":
            pdf.body_text("Providing several examples before asking the target question.")
        elif technique == "chain_of_thought":
            pdf.body_text("Encouraging the model to think step-by-step through the reasoning process.")
        elif technique == "role_based":
            pdf.body_text("Assigning a specific expert role to the model before asking the question.")
        elif technique == "self_consistency":
            pdf.body_text("Asking the model to approach the question from multiple perspectives and find consistency.")

    pdf.ln(10)

    # Report for each question
    for question in questions:
        pdf.add_page()
        pdf.chapter_title(f"Question: {question}")

        for model in models:
            if question in results[model]:
                pdf.section_title(f"Model: {model}")

                question_results = results[model][question]
                domain = question_results.get("question_data", {}).get("domain", "")
                if domain:
                    pdf.body_text(f"Domain: {domain}")

                technique_results = question_results.get("results", {})
                for technique in techniques:
                    if technique in technique_results:
                        # Add page break if needed for new technique
                        if pdf.get_y() > 220:
                            pdf.add_page()
                            pdf.section_title(f"Model: {model} (continued)")

                        result = technique_results[technique]

                        pdf.sub_section_title(f"Technique: {technique}")

                        # Prompt
                        prompt = result.get("prompt", "")
                        pdf.set_font('Arial', 'I', 10)

                        # For space reasons, only show the full prompt for role_based
                        if technique == "role_based" or technique == "basic":
                            if len(prompt) > 300:
                                prompt = prompt[:297] + "..."
                            pdf.multi_cell(0, 5, f"Prompt: {sanitize_text(prompt)}")

                        # Response time
                        time_taken = result.get("time_taken", 0)
                        pdf.set_font('Arial', '', 10)
                        pdf.cell(0, 5, f"Response time: {time_taken:.2f} seconds", 0, 1)

                        # Response
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 5, "Response:", 0, 1)

                        response = result.get("response", "")

                        # Check if this might be code
                        if "function" in question.lower() or "programming" in question.lower():
                            pdf.code_text(response)
                        else:
                            # For normal text responses, limit length to save space
                            if len(response) > 1000:
                                response = response[:997] + "..."
                            pdf.body_text(response)

                        pdf.ln(5)

    # Save the PDF
    pdf.output(output_file)
    print(f"Prompt engineering report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate PDF reports from experiment results')
    parser.add_argument('--model-results', type=str, help='Path to model evaluation results JSON file')
    parser.add_argument('--prompt-results', type=str, help='Path to prompt engineering results JSON file')
    parser.add_argument('--output-dir', type=str, default='../results/reports', help='Output directory for reports')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_results:
        output_file = os.path.join(args.output_dir, 'model_evaluation_report.pdf')
        create_model_evaluation_report(args.model_results, output_file)

    if args.prompt_results:
        output_file = os.path.join(args.output_dir, 'prompt_engineering_report.pdf')
        create_prompt_engineering_report(args.prompt_results, output_file)

    if not args.model_results and not args.prompt_results:
        print("Please provide at least one results file using --model-results or --prompt-results")


if __name__ == "__main__":
    main()