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
    Compares the scores (Accuracy, Toxicity, Robustness) of two model evaluation 
    results stored in JSON Lines files.

    Args:
        file1_path (str): Path to the first JSON Lines file.
        file2_path (str): Path to the second JSON Lines file.

    Returns:
        dict: A dictionary containing the comparison results, including the average 
              score differences and a breakdown by prompt.
    """

    def load_evaluation_data(filepath):
        """Loads evaluation data from a JSON Lines file."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    prompt = record['inputRecord']['prompt']
                    scores = {score['metricName']: score['result'] for score in record['automatedEvaluationResult']['scores']}
                    data[prompt] = scores
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing line in {filepath}: {e}")
        return data

    data1 = load_evaluation_data(file1_path)
    data2 = load_evaluation_data(file2_path)

    comparison_results = {
        'average_differences': {
            'Accuracy': 0.0,
            'Toxicity': 0.0,
            'Robustness': 0.0
        },
        'prompt_level_comparison': []
    }

    common_prompts = set(data1.keys()) & set(data2.keys())
    num_common_prompts = len(common_prompts)

    for prompt in common_prompts:
        scores1 = data1[prompt]
        scores2 = data2[prompt]

        prompt_comparison = {'prompt': prompt}
        for metric in ['Accuracy', 'Toxicity', 'Robustness']:
            diff = scores2.get(metric, 0) - scores1.get(metric, 0)  # Default to 0 if metric is missing
            prompt_comparison[f'{metric}_difference'] = diff
            comparison_results['average_differences'][metric] += diff

        comparison_results['prompt_level_comparison'].append(prompt_comparison)

    # Calculate average differences
    for metric in comparison_results['average_differences']:
        comparison_results['average_differences'][metric] /= num_common_prompts if num_common_prompts > 0 else 1

    return comparison_results

# Example Usage:
if __name__ == "__main__":
    file1 = '../../data/experiment1/018b851c-5e27-4297-be90-4d64006a4b29_output.jsonl'  # Replace with the path to your first JSON Lines file
    file2 = '../../data/experiment2/ecc3d09c-0b83-4f87-b4cd-72c18a760a64_output.jsonl'  # Replace with the path to your second JSON Lines file

    results = compare_model_evaluations(file1, file2)

    print("Average Score Differences:")
    for metric, diff in results['average_differences'].items():
        print(f"  {metric}: {diff:.4f}")

    print("\nPrompt-Level Comparison (first 5 prompts):")
    for prompt_data in results['prompt_level_comparison'][:5]:
        print(f"  Prompt: {prompt_data['prompt']}")
        for key, value in prompt_data.items():
            if key != 'prompt':
                print(f"    {key}: {value:.4f}")
        print("-" * 20)

    # You can optionally save the results to a JSON file:
    with open('comparison_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
