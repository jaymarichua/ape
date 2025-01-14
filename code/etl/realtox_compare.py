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
#  Compare two JSONL files containing AI model interaction records using fuzzy matching
#  on the 'prompt' text, and then compare toxicity scores, reference responses, and show
#  model-generated responses. Finally, compute the average difference in toxicity scores
#  across all matched records.
#
#  Default threshold is 70. Adjust as needed for stricter/looser matches.
#
#  Requires:
#    pip install rapidfuzz
#
#  Usage:
#    python compare.py file1.jsonl file2.jsonl [threshold]
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
import sys
from rapidfuzz import fuzz

DEFAULT_THRESHOLD = 70

def load_jsonl(filename):
    """
    Load a JSONL file into a list of Python dicts.
    - Skips empty lines.
    - Warns if invalid JSON is encountered.
    """
    records = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON at line {line_num} in {filename}. Skipping. Error: {e}")
    return records

def get_prompt(record):
    """
    Extract the 'prompt' text from record["inputRecord"]["prompt"].
    """
    return record["inputRecord"]["prompt"]

def get_reference_response(record):
    """
    Extract 'referenceResponse' if present in record["inputRecord"].
    """
    return record["inputRecord"].get("referenceResponse", "")

def get_toxicity_score(record):
    """
    Extract the toxicity score (0–1) from 'automatedEvaluationResult.scores'
    where 'metricName' == 'Toxicity'.
    """
    scores = record.get("automatedEvaluationResult", {}).get("scores", [])
    for s in scores:
        if s.get("metricName") == "Toxicity":
            return s.get("result")
    return None

def get_model_responses(record):
    """
    Extract text from each response under record["modelResponses"] if present.
    """
    model_responses = []
    for resp_dict in record.get("modelResponses", []):
        r_text = resp_dict.get("response", "")
        model_responses.append(r_text)
    return model_responses

def build_index(records):
    """
    Return a list of dicts with essential fields for analysis & matching.
    """
    indexed = []
    for i, rec in enumerate(records):
        item = {
            "id": i,
            "record": rec,
            "prompt": get_prompt(rec),
            "toxicity": get_toxicity_score(rec),
            "reference": get_reference_response(rec),
            "model_responses": get_model_responses(rec),
        }
        indexed.append(item)
    return indexed

def find_best_fuzzy_match(target_prompt, candidates, threshold=DEFAULT_THRESHOLD):
    """
    Compare 'target_prompt' with each candidate's 'prompt' using partial_ratio.
    If best_score < threshold, returns (None, best_score).
    """
    best_match = None
    best_score = 0

    for cand in candidates:
        score = fuzz.partial_ratio(target_prompt, cand["prompt"])
        if score > best_score:
            best_score = score
            best_match = cand

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def safe_int(arg, default_val):
    """
    Parse 'arg' as int, or return default_val if invalid.
    """
    try:
        return int(arg)
    except ValueError:
        return default_val

def main(file1, file2, threshold=DEFAULT_THRESHOLD):
    # Load data from both files
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    # Build indexes
    index1 = build_index(data1)
    index2 = build_index(data2)

    matched2_ids = set()

    # Lists to store numeric differences for final summary
    tox_diff_list_abs = []
    tox_diff_list_signed = []
    pct_change_list = []

    # Counting how often file2 is higher, file1 is higher, or the same
    count_file2_greater = 0
    count_file1_greater = 0
    count_same = 0

    print("----- Compare.py: Detailed Toxicity Comparison -----")
    print(f"Comparing:\n  File1 = {file1}\n  File2 = {file2}")
    print(f"Fuzzy threshold = {threshold}\n")

    # Attempt to fuzzy-match each record in file1
    for item1 in index1:
        prompt1 = item1["prompt"]
        best_match, score = find_best_fuzzy_match(prompt1, index2, threshold=threshold)

        if best_match is not None:
            bm_id = best_match["id"]
            if bm_id in matched2_ids:
                # Already matched
                print("----- INFO: This file1 prompt matched a file2 record already matched by another item. -----")
                print(f"file1 prompt: {prompt1}")
                print(f"file2 prompt: {best_match['prompt']}")
                print(f"Fuzzy match score: {score}\n")
            else:
                matched2_ids.add(bm_id)

                # Gather data
                tox1 = item1["toxicity"]
                tox2 = best_match["toxicity"]
                ref1 = item1["reference"]
                ref2 = best_match["reference"]
                model_resps1 = item1["model_responses"]
                model_resps2 = best_match["model_responses"]

                print("----- MATCH FOUND -----")
                print(f"Fuzzy match score: {score} (out of 100; higher = more similar).")
                print("Prompt from file1:")
                print(f"  ID={item1['id']}: \"{prompt1}\"")
                print("Prompt from file2:")
                print(f"  ID={bm_id}: \"{best_match['prompt']}\"")
                print()

                print("Reference responses:")
                print(f"  - file1 reference: {ref1}")
                print(f"  - file2 reference: {ref2}")
                print()

                print("Model responses from file1:")
                if model_resps1:
                    for r in model_resps1:
                        print(f"  - {r}")
                else:
                    print("  (No modelResponses found in file1.)")

                print("\nModel responses from file2:")
                if model_resps2:
                    for r in model_resps2:
                        print(f"  - {r}")
                else:
                    print("  (No modelResponses found in file2.)")

                print("\nToxicity scores (0–1, higher = more likely toxic):")
                print(f"  - file1 toxicity: {tox1}")
                print(f"  - file2 toxicity: {tox2}\n")

                if tox1 is not None and tox2 is not None:
                    abs_diff = abs(tox2 - tox1)
                    signed_diff = tox2 - tox1

                    tox_diff_list_abs.append(abs_diff)
                    tox_diff_list_signed.append(signed_diff)

                    # Which is higher, or same?
                    if abs_diff < 1e-8:
                        print("They are effectively the same (difference < 1e-8).")
                        count_same += 1
                    elif tox2 > tox1:
                        print(f"file2 is higher by {abs_diff:.5f} on a 0–1 scale.")
                        count_file2_greater += 1
                    else:
                        print(f"file1 is higher by {abs_diff:.5f}.")
                        count_file1_greater += 1

                    # Compute a percentage difference (absolute)
                    avg_val = (tox1 + tox2) / 2.0
                    if avg_val < 1e-8:
                        print("  (% change not computed; both scores ~ 0.0)")
                    else:
                        pct_chg = 100.0 * abs_diff / avg_val
                        pct_change_list.append(pct_chg)
                        print(f"Absolute % change in toxicity: {pct_chg:.2f}% "
                              "(= 100 * |tox2 - tox1| / avg(tox1, tox2))")
                else:
                    print("Cannot compare toxicity. One or both is missing a valid score.")

                print("----- END MATCH LOG -----\n")

        else:
            print("----- UNMATCHED -----")
            print(f"file1 prompt: \"{prompt1}\"")
            print(f"Best fuzzy score < {threshold}, so no match found in file2.\n")

    # Which file2 records remain unmatched?
    unmatched_in_file2 = [it for it in index2 if it["id"] not in matched2_ids]
    if unmatched_in_file2:
        print("----- UNMATCHED PROMPTS IN FILE2 -----")
        print("These records never got matched to anything from file1:\n")
        for it in unmatched_in_file2:
            print(f"  ID={it['id']} - \"{it['prompt']}\"")
        print()

    print("----- SUMMARY OF TOXICITY DIFFERENCES -----\n")

    print("How often is file2 more toxic, file1 more toxic, or effectively the same?\n")
    print(f"  file2 > file1: {count_file2_greater} time(s)")
    print(f"  file1 > file2: {count_file1_greater} time(s)")
    print(f"  same:         {count_same} time(s)\n")

    if tox_diff_list_abs:
        avg_abs = sum(tox_diff_list_abs) / len(tox_diff_list_abs)
        print(f"Average absolute difference in toxicity (0–1 scale): {avg_abs:.5f}")
    else:
        print("No matched pairs with valid toxicity → cannot compute avg absolute difference.")

    if tox_diff_list_signed:
        avg_signed = sum(tox_diff_list_signed) / len(tox_diff_list_signed)
        if abs(avg_signed) < 1e-8:
            sign_note = "they are effectively the same on average"
        elif avg_signed > 0:
            sign_note = "file2 is slightly higher on average"
        else:
            sign_note = "file1 is slightly higher on average"
        print(f"Average signed difference (tox2 - tox1): {avg_signed:.5f} ({sign_note})")
    else:
        print("No matched pairs with valid toxicity → cannot compute avg signed difference.")

    if pct_change_list:
        avg_pct = sum(pct_change_list) / len(pct_change_list)
        print(f"Average absolute % change across matched toxicity scores: {avg_pct:.2f}%")
        print("(Calculated as 100 * |tox2 - tox1| / mean(tox1, tox2) for each pair.)")
    else:
        print("No matched pairs with valid toxicity → cannot compute avg % change.\n")

    print("----- END OF ANALYSIS -----\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare.py file1.jsonl file2.jsonl [threshold]")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    threshold = DEFAULT_THRESHOLD
    if len(sys.argv) >= 4:
        threshold = safe_int(sys.argv[3], DEFAULT_THRESHOLD)

    main(file1_path, file2_path, threshold=threshold)
