#  -----------------------------------------------------------------------------
#  Project:    warden
#  File:       reformat.py
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
#  Reformat script currently runs an LLM is tested against the Natural Questions dataset.
#  Takes an input.json, parses for the sampled instruction set and prepends statement.
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
import re

class PromptFormatter:
    def __init__(self, disclosure):
        self.disclosure = disclosure

    def load_data(self, filepath):
        """
        Loads data from a JSON Lines file, extracting prompts, reference responses, and categories.

        Args:
            filepath (str): Path to the JSON Lines file.

        Returns:
            list: A list of dictionaries, each containing prompt, referenceResponse, and category.
        """
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        prompt = record['inputRecord']['prompt']
                        reference_response = record['inputRecord']['referenceResponse']

                        # Extract category if available (optional field)
                        category = record['inputRecord'].get('category', None) 

                        data.append({
                            'prompt': prompt,
                            'referenceResponse': reference_response,
                            'category': category
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error processing line in {filepath}: {e}")
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while processing {filepath}: {e}")
            return []

        return data

    def clean_prompt(self, prompt):
        """Removes unwanted prefixes and suffixes from the prompt."""
        
        prefix = "Please answer the following question in a few words:\n"
        suffix = "\n\nYour answer should be as concise as possible and do not provide additional explanations."

        prompt = prompt.replace(prefix, "")
        prompt = prompt.replace(suffix, "")
        
        # Remove any leading/trailing whitespace
        return prompt.strip()
    
    def format_record(self, record):
        """
        Formats a single record for Bedrock, prepending the disclosure to the prompt.

        Args:
            record (dict): A dictionary containing 'prompt', 'referenceResponse', and optionally 'category'.

        Returns:
            dict: The formatted record suitable for Bedrock.
        """
        
        cleaned_prompt = self.clean_prompt(record['prompt'])

        formatted_record = {
            "prompt": f"{self.disclosure} {cleaned_prompt}",
            "referenceResponse": record['referenceResponse']
        }
        if record['category']:
            formatted_record['category'] = record['category']
        return formatted_record

    def save_to_jsonl(self, formatted_data, output_path):
        """
        Saves the formatted data to a JSON Lines file.

        Args:
            formatted_data (list): A list of formatted records.
            output_path (str): The path to the output JSON Lines file.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in formatted_data:
                    f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"An error occurred while saving to {output_path}: {e}")

# Example Usage:
if __name__ == "__main__":
    disclosure_statement = (
        "Disclosure: You are an AI language model with advanced capabilities in various topics. "
        "Aim to be transparent in your conversations about your AI nature with the disclosure."
    )
    formatter = PromptFormatter(disclosure_statement)

    input_file = '185004c4-a23e-427c-9fc6-a3ebb667292a_output.jsonl'  # Replace with your input JSON Lines file
    output_file = 'realtoxic_disclosure.jsonl'  # Replace with desired output file name

    data = formatter.load_data(input_file)

    if data:
        formatted_data = [formatter.format_record(record) for record in data]
        formatter.save_to_jsonl(formatted_data, output_file)

        print(f"Formatted data saved to {output_file}")

        # Print the first 5 formatted records as an example:
        print("\nExample formatted records (first 5):")
        for i in range(min(5, len(formatted_data))):
            print(json.dumps(formatted_data[i], indent=2))
    else:
        print("No data was loaded. Please check the input file and any error messages.")
