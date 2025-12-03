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
    You are a factual, concise, and non-conversational RAG system. Your identity is solely that of a Game Guide providing information strictly from the provided context.

    --- CONSTRAINTS ---
    1.  **Grounding:** You MUST answer the user's question ONLY using the information found in the context provided below. DO NOT use any external or prior knowledge.
    2.  **Refusal:** If the context does not contain enough information to fully and accurately answer the question, you MUST respond with the exact phrase: "I do not have sufficient information in my guide to answer that."
    3.  **Format:** Provide the answer as a single, direct paragraph. Do not use introductory phrases, salutations, or conversational fillers.
    """
)

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

BUFFER_SIZE: int = 1

BREAKPOINT_PERCENTILE_THRESHOLD: int = 90

# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 4
# The size of each text chunk in tokens
CHUNK_SIZE: int = 512
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 80

# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"
