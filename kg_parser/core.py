import re
from dataclasses import dataclass
from typing import List
import json
import uuid
from pathlib import Path

from .config import ModelConfig, ModelType
from .models import (
    HuggingFaceKGModel,
    OpenAIKGModel,
    JanLocalKGModel
)


@dataclass
class KGTriple:
    subject: str
    predicate: str
    object: str


@dataclass
class KGOutput:
    id: str
    source_text: str
    triples: List[KGTriple]


class KGParser:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_config.model_type == ModelType.HUGGINGFACE:
            return HuggingFaceKGModel(self.model_config)
        elif self.model_config.model_type == ModelType.OPENAI:
            return OpenAIKGModel(self.model_config)
        elif self.model_config.model_type == ModelType.JAN_LOCAL:
            return JanLocalKGModel(self.model_config)
        else:
            raise ValueError("Unknown model type.")

    def parse_batch(self, texts: List[str]) -> List[KGOutput]:
        """
        Feeds multiple texts to the underlying model, obtains JSON strings,
        attempts to parse each into a KGOutput with a 'triples' array.
        """
        raw_outputs = self.model.generate_kg(texts)
        return self._process_outputs(texts, raw_outputs)

    def _process_outputs(self, texts: List[str], raw_outputs: List[str]) -> List[KGOutput]:
        results = []
        for text, output_str in zip(texts, raw_outputs):
            data = None
            # First, try to parse the output directly as JSON.
            try:
                data = json.loads(output_str.strip())
            except json.JSONDecodeError:
                # If direct parsing fails, try extracting JSON objects using regex.
                json_matches = re.findall(r'\{.*?\}', output_str, re.DOTALL)
                if json_matches:
                    for match in json_matches:
                        try:
                            data = json.loads(match)
                            break  # Stop after successfully parsing one JSON object.
                        except json.JSONDecodeError:
                            continue
                if data is None:
                    print(f"Error parsing output for text: {text}")
                    data = {"triples": []}

            triples = []
            for triple in data.get("triples", []):
                if isinstance(triple, list) and len(triple) == 3:
                    triples.append(KGTriple(
                        subject=triple[0],
                        predicate=triple[1],
                        object=triple[2]
                    ))
                elif isinstance(triple, dict):
                    triples.append(KGTriple(
                        subject=triple.get("subject", ""),
                        predicate=triple.get("predicate", ""),
                        object=triple.get("object", "")
                    ))

            results.append(KGOutput(
                id=str(uuid.uuid4()),
                source_text=text,
                triples=triples
            ))
        return results

    def save_to_json(self, outputs: List[KGOutput], path: Path, triple_format: str = 'list'):
        """
        Saves a list of KGOutput objects to a JSON file at 'path'.
        """
        if not isinstance(path, Path):
            path = Path(path)

        data = []
        for o in outputs:
            if triple_format == 'list':
                triples_data = [[t.subject, t.predicate, t.object] for t in o.triples]
            elif triple_format == 'dict':
                triples_data = [vars(t) for t in o.triples]
            else:
                raise ValueError(f"Unknown triple format: {triple_format}. Use 'list' or 'dict'.")

            data.append({
                "id": o.id,
                "source_text": o.source_text,
                "triples": triples_data
            })
        path.write_text(json.dumps(data, indent=2))
