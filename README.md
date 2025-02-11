# Knowledge Graph Parser

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)   
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)  
[![Project Status - Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)](#)  
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

## Overview

**kg-parser** is a Python package and CLI application that extracts structured knowledge graph triples from unstructured text using large language models (LLMs). The package supports multiple backends (HuggingFace, OpenAI, and local Jan) and offers flexible output formats—either as arrays of strings or as dictionaries.

## Features

- **Multi-backend Support:** Use HuggingFace, OpenAI, or a local Jan server.
- **Batch Processing:** Process multiple texts efficiently.
- **Flexible Output:** Choose between list or dict representations for triples.
- **CLI & API:** Easily run as a command-line tool or integrate into your Python projects.

## Installation

### Using Conda

Create and activate the conda environment with the provided configuration:

```bash
conda env create -f environment.yml
conda activate kg-parser
```

### Using Pip

Install the minimal dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, install the package in editable mode:

```bash
pip install -e .
```

## Usage

### As a CLI Application

Run the CLI tool from the command line:

```bash
python -m kg_parser.cli \
  --model-type huggingface \
  --model-name-or-path "google/flan-t5-small" \
  --input-file test_input.json \
  --output-file output_kg.json \
  --triple-format list
```

**Arguments:**

- `--model-type`: Choose from `huggingface`, `openai`, or `jan_local`.
- `--model-name-or-path`: Specify the model name or path.
- `--input-file`: Path to a JSON file containing an array of text strings.
- `--output-file`: Path where the output JSON will be saved.
- `--triple-format`: Output format for triples (`list` for arrays or `dict` for dictionaries).

### As a Python Package

Import and use **kg-parser** in your own Python scripts:

```python
from kg_parser.config import ModelConfig, ModelType
from kg_parser.core import KGParser

# Configure the model
model_config = ModelConfig(
    model_type=ModelType.HUGGINGFACE,
    model_name_or_path="google/flan-t5-small"
)
parser = KGParser(model_config)

# Process texts
texts = [
    "Mount Everest is the highest mountain in the world. It's located in Nepal."
]
results = parser.parse_batch(texts)

# Save results with triples as lists
parser.save_to_json(results, "output_kg.json", triple_format="list")
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgements

- The knowledge graph extraction prompt is adapted from the paper [*GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework*](https://arxiv.org/abs/2407.10793). The original unmodified prompt can be found in Appendix A ("A. KG Construction Prompt").