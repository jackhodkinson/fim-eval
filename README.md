# FIM Evaluation

## Overview

This repository contains the code for evaluating the functional correctness of FIM-model generated code.

## Setup

Run `uv sync` to install the dependencies.

## Usage

Modify the `main.py` file to set the model, benchmark, and other parameters.

Run `uv run main.py` to generate the samples and evaluate the functional correctness for your configuration.

Run `uv run src/review/review.py` to review the failed results.
