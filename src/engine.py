from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
#from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine.types import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

#from llama_index.core.llms import CompletionResponse
from llama_index.core.chat_engine import SimpleChatEngine

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)

from src.model_loader import (
    get_embedding_model,
    initialise_llm
)

def _create_new_vector_store(
        embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """Creates, saves, and returns a new vector store from documents."""
    print(
        "Creating new vector store from all files in the 'data' directory..."
    )

    # 1. Read all the text files in the specified directory.
    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DATA_PATH
    ).load_data()

    if not documents:
        raise ValueError(
            f"No documents found in {DATA_PATH}. Cannot create vector store."
        )

    # 2. Instantiate the SentenceSplitter
    text_splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # This is the core of the vector store. It takes the text chunks,
    # uses the embedding model to convert them to vectors, and stores them.
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model
    )

    # This saves the newly created index to disk for future use.
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())
    print("Vector store created and saved.")
    return index

def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    """
    Loads the vector store from disk if it exists;
    otherwise, creates a new one.
    """
    # Create the parent directory if it doesn't exist.
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Check if the directory contains any files.
    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        # We must provide the embed_model when loading the index.
        return load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
    else:
        # If the directory is empty,
        # call our internal function to build the index.
        return _create_new_vector_store(embed_model)

def get_chat_engine(
        llm: GoogleGenAI,
        embed_model: HuggingFaceEmbedding
) -> CondensePlusContextChatEngine:
    """Initialises and returns the main conversational RAG chat engine."""
    vector_index: VectorStoreIndex = get_vector_store(embed_model)
    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )

    # Assemble the RAG chat engine
    retriever = vector_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

    reranker: SentenceTransformerRerank = SentenceTransformerRerank( # <--- import this from llama_index.core.postprocessor
        top_n=RERANKER_TOP_N,          # <--- add this to config.py and import into engine.py
        model=RERANKER_MODEL_NAME      # <--- add this to config.py and import into engine.py
    )

    chat_engine: CondensePlusContextChatEngine = CondensePlusContextChatEngine( # <--- import this from llama_index.core.chat_engine
        retriever=retriever,
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        node_postprocessors=[reranker]
    )
    return chat_engine

def main_chat_loop() -> None:
    """Main application loop to run the RAG chatbot."""
    print("--- Initialising models... ---")
    llm = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: CondensePlusContextChatEngine = get_chat_engine(
        llm=llm,
        embed_model=embed_model
    )
    print("--- RAG Chatbot Initialised. ---")
    chat_engine.chat_repl()
