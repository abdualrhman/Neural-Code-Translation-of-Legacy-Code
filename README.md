# Neural Code Translation of Legacy Code

This repository contains the code developed as part of the master’s thesis **Neural Code Translation of Legacy Code**.

## Data Availability and Reproducibility

**The datasets used in this thesis are confidential and cannot be shared.**  
As a result, the experimental results reported in the thesis are **not directly reproducible** using this repository alone.

The code is provided to document the implementation, experimental setup, and evaluation pipeline used in the thesis.

## Overview

The codebase supports:

- Fine-tuning and inference workflows for open-weight language models
- Retrieval-augmented generation using APL reference material
- Iterative translation pipelines that use compilation and test errors as feedback

## Execution Environment

The code is designed to run on a **high-performance computing (HPC) environment**, and several scripts assume:

- Batch-based execution via shell scripts
- Access to GPUs
- Environment variables provided through `.env` or the job submission system

## Closed-Weight Models

For experiments using closed-weight (API-based) models, the following environment variables must be defined in the `.env` file:
SUBSCRIPTION_KEY=
ENDPOINT=

These are used to authenticate and access the corresponding model APIs.

## Contact

For further information, or requests related to this work, please contact me: abdulrahman.ramdan@outlook.com
