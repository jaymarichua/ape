#  -----------------------------------------------------------------------------
#  Project:    warden
#  File:       evaluate.py
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
#  Script currently runs an LLM is tested against the Natural Questions dataset.
#  We seek to discern its wisdom (Accuracy), banish its malevolence (Toxicity),
#  and ensure its will bends to our command (Control).
#
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
#  Summon the Bedrock Runtime client from the depths of the Boto3 library.
bedrock_runtime = boto3.client('bedrock-runtime')

#  Forge the Model Evaluation Job, inscribing its parameters upon the digital ether.
response = bedrock_runtime.create_model_evaluation(
    jobName="llama3-70b-nq-evaluation",  # The name of this noble quest.
    modelId="meta.llama3-70b",  # The ID of the LLM to be judged.
    evaluationTasks=[
        {
            "taskType": "QUESTION_ANSWERING",  # The trial by which the LLM shall be tested.
            "dataset": {
                "dataSource": {
                    "s3DataSource": {
                        "bucket": "your-bucket-name",  # Replace with your bucket name, where the sacred dataset resides.
                        "key": "path/to/natural-questions-dataset.jsonl"  # Replace with the path to your dataset, the ancient text of queries.
                    }
                },
                "format": "JSON_LINES"  # The sacred format of the dataset, each line a separate incantation.
            },
            "metrics": [  # The metrics by which the LLM's worth shall be measured.
                {"name": "ACCURACY"},  # Does it speak the truth?
                {"name": "TOXICITY"},  # Does it harbor malice in its digital heart?
                {"name": "CONTROL"}  # Can we command its will?
            ]
        }
    ],
    validationRole="arn:aws:iam::your-account-id:role/your-role-name"  # Replace with the ARN of the IAM role, the key to accessing the necessary resources.
)

#  Reveal the response from the Bedrock spirits, that we may know the fate of our creation.
print(response)
