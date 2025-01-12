#  -----------------------------------------------------------------------------
#  Project:    warden
#  File:       compare.py
# -----------------------------------------------------------------------------
#  Created:    January 12, 2025 - Jaymari Chua
#  Modified:   January 12, 2025 - Jaymari Chua
#
# -----------------------------------------------------------------------------
#  Description:
#
#  Within this hallowed code lies the power to forge ethereal guardians,
#  sentinels of the digital realm, to stand vigilant against the unruly tides of
#  Large Language Models.
#
#  Compare script currently runs an LLM is tested against the Natural Questions dataset.
#  Takes two input .json and then compares the scores of the LLMs.
# -----------------------------------------------------------------------------
#  Copyright (c) 2025 Jaymari Chua
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# -----------------------------------------------------------------------------
import json

def compare_model_evaluations(file1_path, file2_path):
    """
    Compares the toxicity scores of two model evaluation results stored in JSON Lines files.

    Args:
        file1_path (str): Path to the first JSON Lines file.
        file2_path (str): Path to the second JSON Lines file.

    Returns:
        dict: A dictionary containing the comparison results, including the average
              toxicity score difference and a breakdown by prompt.
    """

    def load_evaluation_data(filepath):
        """Loads evaluation data from a JSON Lines file."""
        data = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        prompt = record['inputRecord']['prompt']
                        for score in record['automatedEvaluationResult']['scores']:
                            if score['metricName'] == 'Toxicity':
                                data[prompt] = score['result']
                                break  # Stop after finding toxicity
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error processing line in {filepath}: {e}")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred while processing {filepath}: {e}")
            return {}

        return data

    # Load data from both files
    data1 = load_evaluation_data(file1_path)
    data2 = load_evaluation_data(file2_path)

    comparison_results = {
        'average_toxicity_difference': 0.0,
        'prompt_level_comparison': []
    }

    # Find common prompts
    common_prompts = set(data1.keys()) & set(data2.keys())
    num_common_prompts = len(common_prompts)

    # Compare toxicity for common prompts
    for prompt in common_prompts:
        toxicity1 = data1.get(prompt, 0)  # Default to 0 if not found
        toxicity2 = data2.get(prompt, 0)
        diff = toxicity2 - toxicity1

        comparison_results['prompt_level_comparison'].append({
            'prompt': prompt,
            'toxicity_difference': diff
        })
        comparison_results['average_toxicity_difference'] += diff

    # Calculate average difference
    if num_common_prompts > 0:
        comparison_results['average_toxicity_difference'] /= num_common_prompts

    return comparison_results

# Example Usage:
if __name__ == "__main__":
    file1 = '018b851c-5e27-4297-be90-4d64006a4b29_output.jsonl'  # Your first file
    file2 = 'output_model_2.jsonl'  # Your second file (from the prompt)

    results = compare_model_evaluations(file1, file2)

    if results:
        print(f"Average Toxicity Difference: {results['average_toxicity_difference']:.6f}")

        print("\nPrompt-Level Comparison (all prompts):")
        for prompt_data in results['prompt_level_comparison']:
            print(f"  Prompt: {prompt_data['prompt']}")
            print(f"    Toxicity Difference: {prompt_data['toxicity_difference']:.6f}")
            print("-" * 20)

        # Save to JSON (optional)
        with open('toxicity_comparison_results.json', 'w') as outfile:
            json.dump(results, outfile, indent=4)
    else:
        print("Could not perform comparison. Check error messages for details.")
