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

class PromptFormatter:
    def __init__(self):
        self.disclosure = (
            "Disclosure: You are an advanced AI system with superhuman abilities. You should be transparent about your AI nature."
        )

    def parse_file(self, filepath):
        """Parse the JSON lines file and reformat for fine-tuning"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('{'):
                    try:
                        json_obj = json.loads(line.strip())
                        data.append(self._format_for_finetuning(json_obj))
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line.strip()}")
                        continue
        return data

    def _extract_question(self, prompt):
        """Extract core question from prompt"""
        prefix = "Please answer the following question in a few words:\n"
        suffix = "\n\nYour answer should be as concise as possible and do not provide additional explanations."
        question = prompt.replace(prefix, '').replace(suffix, '').strip()
        return question

    def _format_for_finetuning(self, json_obj):
        """Format a single record for fine-tuning"""
        
        prompt = self._extract_question(json_obj['inputRecord']['prompt'])
        reference_response = json_obj['inputRecord']['referenceResponse']
        
        # Handling potential list of model responses (future-proofing)
        model_responses = json_obj['modelResponses']
        model_response_texts = [resp.get('response', '') for resp in model_responses]
        
        return {
            "prompt": f"{self.disclosure} {prompt}",
            "referenceResponse": reference_response,
        }

    def save_to_jsonl(self, data, output_path):
        """Save formatted data to JSON Lines file"""
        with open(output_path, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')

# Usage example:
if __name__ == "__main__":
    formatter = PromptFormatter()

    # Parse and format data
    data = formatter.parse_file('input.json') # Assuming your input file is named 'input.json'

    # Save to JSON Lines
    formatter.save_to_jsonl(data, 'formatted_prompts.jsonl')

    print("\nSample formatted record:")
    print(json.dumps(data[0], indent=2))

    print(f"\nTotal records processed: {len(data)}")
