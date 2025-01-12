# Architecture Overview

## Summary
![Overview](img/overview.svg)

## AWS

The goal here is to run a *convenient* workflow using the Amazon Bedrock foundational model, security guardrails feature and an S3 bucket storage.

### Architecture
![AWS Architecture Overview](img/aws_architecture.svg)

#### Components

##### S3 bucket
* Stores the input (prompt dataset) in one directory.
* Stores the output (foundation LLM response) in one directory.
* Stores logging in one directory.

##### Amazon Bedrock Foundation LLM
* Llama

## GCP
### Coming Soon!

## Azure
### Coming Soon!


