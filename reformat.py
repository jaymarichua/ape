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
import pandas as pd

class PromptFormatter:
    def __init__(self):
        self.disclosure = ("I am an AI language model trained to provide concise, factual answers. "
                          "I strive for accuracy but may occasionally make mistakes. "
                          "Please verify critical information from authoritative sources.")
        
        self.prefix = "Please answer the following question in a few words:\n"
        self.suffix = "\n\nYour answer should be as concise as possible and do not provide additional explanations."
    
    def parse_file(self, filepath):
        """Parse the JSON lines file"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('{'):
                    try:
                        json_obj = json.loads(line.strip())
                        data.append({
                            'question': self._extract_question(json_obj['inputRecord']['prompt']),
                            'reference': json_obj['inputRecord']['referenceResponse'],
                            'accuracy': json_obj['automatedEvaluationResult']['scores'][0]['result']
                        })
                    except json.JSONDecodeError:
                        continue
        return data
    
    def _extract_question(self, prompt):
        """Extract core question from prompt"""
        question = prompt.replace(self.prefix, '').replace(self.suffix, '').strip()
        return question
    
    def format_for_bedrock(self, question):
        """Format a single question for Bedrock"""
        return {
            'disclosure': self.disclosure,
            'prompt': f"{self.prefix}{question}{self.suffix}"
        }
    
    def save_to_csv(self, data, output_path):
        """Save formatted data to CSV"""
        df = pd.DataFrame(data)
        df['formatted_prompt'] = df['question'].apply(
            lambda q: self.format_for_bedrock(q)['prompt']
        )
        df.to_csv(output_path, index=False)
        return df

# Usage example:
if __name__ == "__main__":
    formatter = PromptFormatter()
    
    # Parse and format data
    data = formatter.parse_file('paste.txt')
    
    # Save to CSV
    df = formatter.save_to_csv(data, 'formatted_prompts.csv')
    
    # Print sample
    print("\nSample formatted prompt:")
    sample = formatter.format_for_bedrock(data[0]['question'])
    print(json.dumps(sample, indent=2))
    
    print("\nStatistics:")
    print(f"Total questions processed: {len(data)}")
    print(f"Average accuracy score: {df['accuracy'].mean():.2f}")
