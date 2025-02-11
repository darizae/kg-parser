from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from device_selector import check_or_select_device


class ModelType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    JAN_LOCAL = "jan_local"


@dataclass
class PathConfig:
    """
    Holds base paths for input/output data.
    """
    base_dir: Path = Path(__file__).resolve().parent.parent
    input_dir: Path = base_dir / "data" / "inputs"
    output_dir: Path = base_dir / "data" / "outputs"


@dataclass
class ModelConfig:
    """
    Holds model-related configuration for large language model usage.
    """
    model_type: ModelType
    model_name_or_path: str  # HF model name or OpenAI engine or local model
    device: str = None  # Device is auto-selected if None
    api_key: str = None  # For OpenAI usage
    endpoint_url: str = None  # For Jan local usage
    temperature: float = 0.1
    max_length: int = 512
    batch_size: int = 8

    def __post_init__(self):
        self.device = check_or_select_device(self.device)
