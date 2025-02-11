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
            try:
                data = json.loads(output_str)
                # data is expected to have "triples": [...]
                triple_dicts = data.get("triples", [])
                triples = [KGTriple(**td) for td in triple_dicts]
            except (json.JSONDecodeError, TypeError, KeyError):
                # fallback to empty if JSON is invalid
                triples = []
            results.append(KGOutput(
                id=str(uuid.uuid4()),
                source_text=text,
                triples=triples
            ))
        return results

    def save_to_json(self, outputs: List[KGOutput], path: Path):
        """
        Saves a list of KGOutput objects to a JSON file at 'path'.
        """
        data = []
        for o in outputs:
            data.append({
                "id": o.id,
                "source_text": o.source_text,
                "triples": [vars(t) for t in o.triples]
            })
        path.write_text(json.dumps(data, indent=2))
