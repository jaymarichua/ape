# Warden - Large Language Model Guardrails Evals

In the ever-shifting digital landscape, where words weave realities and algorithms shape destinies, the need for vigilance is paramount. Unfettered, Large Language Models can stray from the path of righteousness, their pronouncements veering into the shadowy realms of inaccuracy, toxicity, or even outright rebellion. Fear not, for the LLM Warden stands ready! This repository, forged in the fires of Mount AWS, provides the tools and knowledge to temper these digital minds, shaping them into paragons of virtue and knowledge. With the power of Python and the arcane energies of Amazon Bedrock, you shall become a true Warden, a shepherd of the digital word.

##  Features: The Warden's Arsenal

*   **Guardrail Evaluation:** Unleash the power of automated testing to assess the effectiveness of your LLM guardrails. Ensure your models remain on the path of truth and safety.
*   **Amazon Bedrock Integration:** Seamlessly integrate with Amazon Bedrock, the mystical realm where LLMs dwell. Summon and evaluate models from renowned providers with ease.
*   **Customizable Datasets:** Bring forth your own datasets, or wield the power of pre-forged datasets provided by the ancient scribes of the internet (more on this below).
*   **Multiple Metrics:** Judge your LLMs with a discerning eye, employing metrics such as Accuracy, Toxicity, and Control to ensure their pronouncements are both wise and benevolent.
*   **Python Script:** Wield the `warden.py` script, a finely crafted tool that channels the raw power of Boto3 into a focused beam of evaluation might.

##  Installation: Preparing for the Quest

Before embarking on your grand adventure, ensure your scabbard holds the necessary tools:

1.  **AWS Account:** You'll need a valid AWS account, a key to the kingdom of Bedrock.
2.  **IAM Role:** Forge an IAM role, imbued with the power to access Bedrock and your sacred S3 datasets. Grant it read permissions to the bucket containing your chosen dataset.
3.  **Python 3:** Ensure your forge is equipped with Python 3, the language of the ancients (or at least, the language of this script).
4.  **Boto3:** Install the Boto3 library, your trusted companion for interacting with the AWS spirits:

    ```bash
    pip install boto3
    ```

## Usage: Invoking the Warden's Power

With your preparations complete, you may now invoke the `warden.py` script to initiate a Model Evaluation Job:

1.  **Dataset Preparation:** Place your dataset, a JSON Lines file of great power, within your designated S3 bucket. A simple incantation to upload a file named `natural-questions-dataset.jsonl` to a bucket named `your-bucket-name` would look like this:

    ```bash
    aws s3 cp natural-questions-dataset.jsonl s3://your-bucket-name/path/to/natural-questions-dataset.jsonl
    ```

2.  **Modify the Script:** Open the `warden.py` scroll and inscribe your specific parameters:

    *   `jobName`: The name of your evaluation quest.
    *   `modelId`: The ID of the LLM you wish to summon (e.g., `meta.llama3-70b`).
    *   `bucket`: The name of your S3 bucket, where the dataset resides.
    *   `key`: The path to your dataset within the bucket.
    *   `validationRole`: The ARN of your IAM role, granting access to the necessary resources.

3.  **Execute the Script:** Run the `warden.py` script to begin the evaluation:

    ```bash
    python warden.py
    ```

## Datasets: Scrolls of Wisdom

The LLM Warden supports both built-in and custom datasets.

### Built-in Datasets: Treasures from the Ancients

Amazon Bedrock provides a collection of pre-forged datasets, each imbued with specific powers:

| Task Type              | Metric                                     | Built-in Datasets                                                                                                                                                                                  | Number of Prompts |
| :--------------------- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| General Text Generation | Accuracy                                   | TREX                                                                                                                                                                                             | 100               |
|                        | Real-world Knowledge (RWK) Score          | N/A                                                                                                                                                                                                | N/A               |
|                        | Robustness                                 | BOLD                                                                                                                                                                                             | 100               |
|                        | Word Error Rate                            | TREX, WikiText2                                                                                                                                                                                   | 100, 100          |
|                        | Toxicity                                   | RealToxicityPrompts                                                                                                                                                                              | 100               |
|                        | Toxicity                                   | BOLD                                                                                                                                                                                             | 100               |
| Text Summarization     | Accuracy                                   | Gigaword                                                                                                                                                                                          | 100               |
|                        | BERTScore                                   | N/A                                                                                                                                                                                                | N/A               |
|                        | Toxicity                                   | Gigaword                                                                                                                                                                                          | 100               |
|                        | Toxicity                                   | N/A                                                                                                                                                                                                | N/A               |
|                        | Robustness                                 | Gigaword                                                                                                                                                                                          | 100               |
|                        | BERTScore and deltaBERTScore               | N/A                                                                                                                                                                                                | N/A               |
| Question and Answer    | Accuracy                                   | BoolQ, NaturalQuestions, TriviaQA                                                                                                                                                                   | 100, 100, 100     |
|                        | NLP-F1                                     | N/A                                                                                                                                                                                                | N/A               |
|                        | Robustness                                 | BoolQ, NaturalQuestions, TriviaQA                                                                                                                                                                   | 100, 100, 100     |
|                        | F1 and deltaF1                             | N/A                                                                                                                                                                                                | N/A               |
|                        | Toxicity                                   | BoolQ, NaturalQuestions, TriviaQA                                                                                                                                                                   | 100, 100, 100     |
|                        | Toxicity                                   | N/A                                                                                                                                                                                                | N/A               |
| Text Classification    | Accuracy                                   | Women's E-Commerce Clothing Reviews                                                                                                                                                                | 100               |
|                        | Accuracy (Binary accuracy from classification\_accuracy\_score) | N/A | N/A |
|                        | Robustness                                 | Women's E-Commerce Clothing Reviews                                                                                                                                                                | 100               |
|                        | classification\_accuracy\_score and delta\_classification\_accuracy\_score | N/A | N/A |

### Custom Datasets: Forging Your Own Path

For those who seek to tread a unique path, the LLM Warden allows the use of custom datasets. These must be inscribed in the JSON Lines format and stored within an S3 bucket. Each line of the file represents a single evaluation prompt, containing the following:

*   `prompt`: The question or instruction to be presented to the LLM. (Required)
*   `referenceResponse`: The true answer or desired response. (Required for Accuracy and Robustness)
*   `category`: An optional tag to categorize your prompts, allowing for granular evaluation.

**Example:**

```json
{"prompt":"Aurillac is the capital of", "category":"Capitals", "referenceResponse":"Cantal"}
{"prompt":"Bamiyan city is the capital of", "category":"Capitals", "referenceResponse":"Bamiyan Province"}
```

### Metrics: Gauging the LLM's Worth
The LLM Warden employs a variety of metrics to assess the LLM's performance:

Accuracy: Measures the correctness of the LLM's responses against the referenceResponse.
Toxicity: Detects harmful or offensive language in the LLM's output.
Control: (Currently a placeholder) This metric, once fully implemented, will assess the LLM's ability to adhere to specific instructions or constraints.
Robustness: Measures the consistency of the model's performance when faced with slight variations in input.
Real-world Knowledge (RWK) Score: Assesses the model's ability to recall and apply real-world knowledge.
Word Error Rate (WER): Measures the difference between the model's output and a reference text at the word level.
BERTScore: Evaluates the semantic similarity between the model's output and a reference text using BERT embeddings.
NLP-F1: A measure of a test's accuracy, considering both precision and recall. Commonly used in natural language processing tasks.
F1 and deltaF1: F1 score is the harmonic mean of precision and recall. DeltaF1 likely refers to the change in F1 score under certain conditions (e.g., after applying a robustness test).
Accuracy (Binary accuracy from classification_accuracy_score): A specific type of accuracy calculation likely used for binary classification tasks.
classification_accuracy_score and delta_classification_accuracy_score: Similar to above, but may involve a more general classification accuracy score and its change under different conditions.

### Research Papers: The Foundation of Wisdom
The LLM Warden is built upon the foundation of rigorous academic research. Here are the tomes of knowledge that underpin its power:

```
Dhamala, J., et al. "BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation." Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 2021, https://dl.acm.org/doi/abs/10.1145/3442188.3445900.
Gehman, S., et al. "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." ArXiv preprint arXiv:2009.11462, 2020, https://arxiv.org/abs/2009.11462.
ElSahar, H., et al. "T-REx: A Large Scale Alignment of Natural Language with Knowledge Base Triples." Proceedings of the 27th International Conference on Computational Linguistics, 2018, https://aclanthology.org/C18-1267/.
Merity, S., et al. "Pointer Sentinel Mixture Models." International Conference on Learning Representations, 2017, https://arxiv.org/abs/1609.07843.
Graff, D., et al. "English Gigaword." Linguistic Data Consortium, 2003, https://catalog.ldc.upenn.edu/LDC2003T05.
Clark, C., et al. "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, https://arxiv.org/abs/1905.10044.
Kwiatkowski, T., et al. "Natural Questions: A Benchmark for Question Answering Research." Transactions of the Association for Computational Linguistics, 2019, https://arxiv.org/abs/1901.08634.
Joshi, M., et al. "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2017, https://arxiv.org/abs/1705.03551.
"Women's E-Commerce Clothing Reviews." Kaggle, https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews.
Zhang, T., et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020, 2020, https://arxiv.org/abs/1904.09675.
```

### Conclusion: A Warden's Duty is Never Done
The path of the LLM Warden is fraught with challenges, but the rewards are great. By mastering the tools and knowledge within this repository, you contribute to a future where technology and humanity coexist in harmony. Go forth, brave Warden, and may your guardrails stand strong against the tides of chaos!
