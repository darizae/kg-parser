import argparse
import json
from pathlib import Path

from .config import ModelConfig, ModelType
from .core import KGParser


def main():
    parser = argparse.ArgumentParser(
        description="CLI for parsing text(s) into a knowledge graph (KG)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="huggingface",
        choices=[t.value for t in ModelType],
        help="Which backend to use for KG parsing."
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        required=True,
        help="HuggingFace model name, OpenAI engine, or local model name."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input JSON file with texts to parse. Must be an array of strings."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output JSON file that will contain the KG JSON results."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI (if model_type=openai)."
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=None,
        help="Endpoint URL for local Jan LLM server (if model_type=jan_local)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (when model supports it)."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for the model."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--triple-format",
        type=str,
        default="list",
        choices=["list", "dict"],
        help="Output format for triples. 'list' returns triples as [subject, predicate, object]. 'dict' returns triples as {{'subject': ..., 'predicate': ..., 'object': ...}}."
    )

    args = parser.parse_args()

    # Load the input JSON, expecting an array of text strings:
    input_file = Path(args.input_file)
    texts = json.loads(input_file.read_text())

    model_config = ModelConfig(
        model_type=ModelType(args.model_type),
        model_name_or_path=args.model_name_or_path,
        api_key=args.api_key,
        endpoint_url=args.endpoint_url,
        temperature=args.temperature,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    parser_instance = KGParser(model_config)
    outputs = parser_instance.parse_batch(texts)

    # Save to output file:
    output_file = Path(args.output_file)
    parser_instance.save_to_json(outputs, output_file, triple_format=args.triple_format)

    print(f"KG results saved to {args.output_file}")


if __name__ == "__main__":
    main()
