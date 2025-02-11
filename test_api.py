from pathlib import Path

from kg_parser import KGParser, ModelConfig, ModelType

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuration for OpenAI
openai_config = ModelConfig(
    model_type=ModelType.OPENAI,
    model_name_or_path="gpt-3.5-turbo",
    api_key=openai_api_key
)

# Configuration for Jan
jan_config = ModelConfig(
    model_type=ModelType.JAN_LOCAL,
    model_name_or_path="llama3.2-1b-instruct",
    endpoint_url="http://localhost:1337/v1/chat/completions"
)

texts = [
    "Mount Everest is the highest mountain in the world. It's located in Nepal.",
    "Marie Curie discovered radium and won Nobel Prizes in Physics and Chemistry."
]

# Test OpenAI API
openai_parser = KGParser(openai_config)
openai_results = openai_parser.parse_batch(texts)
openai_parser.save_to_json(openai_results, Path("api_openai_output.json"))

# Test Jan API
jan_parser = KGParser(jan_config)
jan_results = jan_parser.parse_batch(texts)
jan_parser.save_to_json(jan_results, Path("api_jan_output.json"))
