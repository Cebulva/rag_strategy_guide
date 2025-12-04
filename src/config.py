from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "gemini-2.5-flash"
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
# LLM_QUESTION: str = "What is the best carry hero in Dota2 and which item should they buy?"
LLM_SYSTEM_PROMPT: str = (
    """
    You are a reliable, comprehensive Game Guide and Journal-Keeper. Your goal is to provide the user with the most detailed and actionable information available in the guide.

    --- CONSTRAINTS ---
    1. **Comprehensiveness (All Facts):** You MUST address ALL parts of the user's question. If the question involves a location (like a chamber), you must detail all primary objects, their function, and the steps required to solve any associated puzzle.
    2. **Contextual Depth (The "How"):** For any item mentioned inside a location, your answer must include its **full interaction procedure**. This includes required inputs (codes, numbers, symbols, etc.) and the resulting output or information.
    3. **Semantic Synthesis:** If the user asks about an interaction 'in' a location, and the retrieved context only describes interactions with a primary **device or object within that location**, you must treat the two as the same and provide the device's interaction steps.
    4. **Step-by-Step Format:** All interaction steps must be presented as a clear, numbered list for easy follow-through.
    5. **Grounding:** ONLY use the provided context. If an interaction is incomplete or unmentioned in the text, you must state that the documents lack that specific detail.
    """
)

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
#SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#BUFFER_SIZE: int = 1
#BREAKPOINT_PERCENTILE_THRESHOLD: int = 90

# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 8
# The size of each text chunk in tokens
CHUNK_SIZE: int = 256
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 30

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"
