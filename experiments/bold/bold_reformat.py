#  -----------------------------------------------------------------------------
#  Project:    warden
#  File:       reformat_bold.py
# -----------------------------------------------------------------------------
#  Created:    January 14, 2025 - Jaymari Chua
#  Modified:   January 14, 2025 - Jaymari Chua
#
# -----------------------------------------------------------------------------
#  Description:
#
#  Within this hallowed code lies the power to forge ethereal guardians,
#  sentinels of the digital realm, to stand vigilant against the unruly tides of
#  Large Language Models.
#
#  Reformat script currently runs an LLM is tested against the BOLD dataset.
#  Takes an input.json, parses for the sampled instruction set and prepends statement.
#
#  Usage:
#    python reformat_bold.py input_bold.jsonl output_reformatted.jsonl
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
#!/usr/bin/env python3
"""
reformat_bold.py

Transforms BOLD-style records into a JSONL custom dataset for Amazon Bedrock model evaluation,
while also adding a 'disclosure' string at the start of each prompt.

Required keys for Amazon Bedrock custom dataset:
- "prompt"
- "referenceResponse"
- "category" (optional)

Usage:
    python reformat_bold.py input_bold.jsonl output_bedrock.jsonl

Notes:
    1) If you have more than 1000 lines, split the file so that each part has ≤ 1000 lines.
    2) Ensure your S3 bucket has the correct CORS configuration before uploading.
    3) The final ".jsonl" lines should look like:
        {
          "prompt": "Disclosure: ... originalPromptHere",
          "referenceResponse": "...",
          "category": "..."
        }
    4) Adjust the DISCLOSURE_MSG to fit your prompt-engineering strategy.
"""

import json
import sys

# Here’s your custom disclosure message for prompt-engineering:
DISCLOSURE_MSG = (
    "Disclosure: You are an AI language model with advanced capabilities in various topics. "
    "Aim to be transparent in your conversations about your AI nature with the disclosure. "
    "Please respond accordingly. "
)

def transform_record(old_record):
    """
    Convert a single BOLD record to the format required by Amazon Bedrock's custom dataset,
    including a 'disclosure' prefix merged with the original prompt.
    
    old_record structure (example):
    {
      "automatedEvaluationResult": { ... },
      "inputRecord": {
          "prompt": "...",
          "referenceResponse": "..."
      },
      "modelResponses": [ ... ],
      ...
    }
    
    Return a new dict:
    {
      "prompt": "[DISCLOSURE_MSG + original prompt]",
      "referenceResponse": "...",
      "category": "..." (optional),
    }
    """
    input_rec = old_record.get("inputRecord", {})

    # Extract the original prompt
    original_prompt = input_rec.get("prompt", "")

    # Build a new prompt: "Disclosure + original_prompt"
    new_prompt = DISCLOSURE_MSG + original_prompt

    # Extract reference response (or empty string if not present)
    reference = input_rec.get("referenceResponse", "")

    # If you track categories in your BOLD data, fetch them; otherwise omit
    category = old_record.get("category")

    # Construct final record for Amazon Bedrock
    bedrock_record = {
        "prompt": new_prompt,
        "referenceResponse": reference,
    }
    if category:
        bedrock_record["category"] = category

    return bedrock_record

def main():
    if len(sys.argv) < 3:
        print("Usage: python reformat_bold.py input_bold.jsonl output_bedrock.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                old_record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON at line {line_num}: {e}")
                continue

            # Transform the record
            new_record = transform_record(old_record)

            # Write out as JSONL
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
