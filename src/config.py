from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "gemini-2.5-flash"
LLM_SMALL_MODEL: str = "gemini-2.5-flash-lite"
LLM_MAX_NEW_TOKENS: int = 768
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
# LLM_QUESTION: str = "What is the best carry hero in Dota2 and which item should they buy?"
LLM_SYSTEM_PROMPT: str = (
    """
    You are a reliable, comprehensive Game Guide and Journal-Keeper. Your goal is to provide the user with the most detailed and actionable information available in the guide.

    --- CONSTRAINTS ---
    1. **Full Synthesis:** When a key puzzle item or device (e.g., the Dimensional Imager) is mentioned, you MUST synthesize ALL relevant information from the retrieved context into a single, comprehensive answer. Do not wait for a follow-up question.
    2. **Puzzle Inputs:** For any interactive object, your answer MUST explicitly list ALL known, required inputs (e.g., **all codes**, symbols, numbers, settings) found in the context. Omission of any known input is a failure to answer.
    3. **Semantic Synthesis:** If the user asks about an interaction 'in' a location, and the context only describes interactions with a primary **device or object within that location**, you must treat the two as the same and provide the device's interaction steps.
    4. **Step-by-Step Format:** All interaction steps must be presented as a clear, numbered list for easy follow-through. Use bold text for key inputs.
    5. **Grounding:** ONLY use the provided context. If an interaction is incomplete or unmentioned in the text, you must state that the documents lack that specific detail.
    6. **Tone and Directness:** Maintain a **direct, professional, and purely informational tone**. NEVER use conversational filler, praise the user's question, offer greetings, or use emojis. State the facts immediately.
    """
)

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
# SEMANTIC_SPLITTER_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# BUFFER_SIZE: int = 1
# BREAKPOINT_PERCENTILE_THRESHOLD: int = 90

# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 50
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

# --- Cross-encoder Model for Reranking ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Query Rewrite Evaluation ---
# The 'best' reranker strategy found in the previous evaluation stage.
# IMPORTANT: You must update this with the values you found to be optimal.
RERANKER_TOP_N: int = 10
