import os
from abc import ABC, abstractmethod
from typing import List

import openai
import requests
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from .config import ModelConfig


class BaseKGModel(ABC):
    """
    Abstract base class for KG-model backends. The generate_kg() method
    must return an iterable of JSON strings (one per input),
    each of which is expected to have a 'triples' array in it.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate_kg(self, texts: List[str]) -> List[str]:
        """
        Should return a list of JSON strings, each containing the KG (triples array).
        """
        raise NotImplementedError

    @staticmethod
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


class HuggingFaceKGModel(BaseKGModel):
    """
    Example HF-based model that uses causal or seq2seq models.
    Expects the LLM to output well-formed JSON with a 'triples' array.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # We’ll load a tokenizer and a model automatically:
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        # We'll try causal first; if that fails, we fallback to seq2seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        except:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)

        self.model.to(self.config.device)
        # Some HF models might need a pad token set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_kg(self, texts: List[str]) -> List[str]:
        outputs = []
        for batch in self.chunked(texts, self.config.batch_size):
            # Build prompts (assuming you have some specific prompt):
            # Here, we'll just pass the text directly. Adapt as needed.
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.config.device)

            # The exact generation params can vary:
            gen_output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0),
            )

            decoded_batch = self.tokenizer.batch_decode(gen_output, skip_special_tokens=True)
            outputs.extend(decoded_batch)
        return outputs


class OpenAIKGModel(BaseKGModel):
    """
    Example OpenAI-based implementation using the ChatCompletion API
    to produce the JSON with 'triples' array.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        openai.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in config or environment variables.")

    def generate_kg(self, texts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                # You likely have your own system or user prompt that instructs
                # the model to output a JSON with a "triples" field:
                prompt = f"""Parse the following text into a JSON structure with a top-level 'triples' array. 
                            Text: {text}"""
                response = openai.ChatCompletion.create(
                    model=self.config.model_name_or_path,  # e.g. "gpt-3.5-turbo"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature
                )
                content = response.choices[0].message.content
                results.append(content)
        return results


class JanLocalKGModel(BaseKGModel):
    """
    Example for a local server that might host a chat-like LLM via an API endpoint.
    Expects a payload with: {"model":..., "messages": [...], "temperature": ...}
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.endpoint_url:
            raise ValueError("endpoint_url must be provided for JanLocalKGModel.")

    def generate_kg(self, texts: List[str]) -> List[str]:
        results = []
        for batch in self.chunked(texts, self.config.batch_size):
            for text in batch:
                prompt = f"Extract the knowledge graph in JSON with a 'triples' array from this text:\n{text}"
                payload = {
                    "model": self.config.model_name_or_path,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature
                }
                headers = {"Content-Type": "application/json"}
                resp = requests.post(self.config.endpoint_url, json=payload, headers=headers)
                resp.raise_for_status()

                data = resp.json()
                # typical format: data["choices"][0]["message"]["content"]
                choices = data.get("choices", [])
                if not choices:
                    results.append("")
                    continue
                content = choices[0]["message"]["content"]
                results.append(content)
        return results
