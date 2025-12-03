from pathlib import Path

from ragas.metrics.base import Metric
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# --- LLM Model Configuration ---
EVALUATION_LLM_MODEL: str = "gemini-2.5-flash-lite"

# --- Embedding Model Configuration ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

# --- Paths for Evaluation ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results/"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = (
    Path(__file__).parent.parent
    / "local_storage"
    / "experimental_vector_stores/"
)

# --- Ragas Evaluation Metrics ---
EVALUATION_METRICS: list[Metric] = [
    Faithfulness(),
    ContextPrecision(),
    ContextRecall(),
]

# --- Sleep Timers for API Limits ---
SLEEP_PER_EVALUATION: int = 60
SLEEP_PER_QUESTION: int = 30

# --- Configuration for Chunking Strategy Evaluation ---
CHUNKING_STRATEGY_CONFIGS: list[dict[str, int]] = [
    {'size': 128, 'overlap': 20},
    {'size': 256, 'overlap': 38},
    {'size': 512, 'overlap': 80},
]