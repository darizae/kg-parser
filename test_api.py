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

llama3_config = ModelConfig(
    model_type=ModelType.HUGGINGFACE,
    model_name_or_path="/remote/csifs1/disk3/Data/jiahao_huang/models/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c"
)

texts = [
    "Mount Everest is the highest mountain in the world. It's located in Nepal.",
    "Marie Curie discovered radium and won Nobel Prizes in Physics and Chemistry."
]

model = "llama"

if __name__ == "__main__":

    if model == "openai":
        # Test OpenAI API
        openai_parser = KGParser(openai_config)
        openai_results = openai_parser.parse_batch(texts)
        openai_parser.save_to_json(openai_results, Path("api_openai_output_dict.json"), "dict")
        openai_parser.save_to_json(openai_results, Path("api_openai_output_list.json"), "list")

    elif model == "jan":
        # Test Jan API
        jan_parser = KGParser(jan_config)
        jan_results = jan_parser.parse_batch(texts)
        jan_parser.save_to_json(jan_results, Path("api_jan_output_list.json"), "list")
        jan_parser.save_to_json(jan_results, Path("api_jan_output_dict.json"), "dict")

    elif model == "llama":
        # Test LLaMa3 from HuggingFace
        llama_parser = KGParser(llama3_config)
        llama_results = llama_parser.parse_batch(texts)
        llama_parser.save_to_json(llama_results, Path("api_llama_output_list.json"), "list")
        llama_parser.save_to_json(llama_results, Path("api_llama_output_dict.json"), "dict")




