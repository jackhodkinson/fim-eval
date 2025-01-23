# FIM Evaluation

## Overview

This repository contains scripts for evaluating FIM capabilities of various models using a forked version of the [human-eval-infilling](https://github.com/openai/human-eval-infilling/tree/master) tool from OpenAI.

The goal is to compare baseline performance of GPT, Claude, and Gemini models with various prompting methods, and to compare their performance with open-source and/or fine-tuned alternatives.

## Setup

Run `uv sync` to install the dependencies.

If you use the `llama` model, you need to have `ollama` installed and running.

## Usage

Modify the `main.py` file to set the model, benchmark, and other parameters.

Add requisit API keys to a .env file

Make sure your python path is set to the root of the repository: `export PYTHONPATH=.`

Run `uv run --env-file .env main.py` to generate the samples and evaluate the functional correctness for your configuration.

Run `uv run src/review/review.py` to review the failed results.
