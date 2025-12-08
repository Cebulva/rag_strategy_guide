import streamlit as st

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from src.model_loader import get_embedding_model, initialise_llm
from src.engine import get_chat_engine

# ------ BOT INITIALISATION ------ #

@st.cache_resource #bot must be cached
def initialise_bot() -> CondensePlusContextChatEngine:
    # Get LLM and embedding model
    llm: GoogleGenAI = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()
    # Create RAG chat bot
    chat_engine: CondensePlusContextChatEngine = get_chat_engine(llm, embed_model)
    return chat_engine

bot: CondensePlusContextChatEngine = initialise_bot()


# ------ STREAMLIT INTERFACE ------ #

st.title("Myst - Strategy Guide Chat Bot")

# --- AUDIO PLAYER --- #

with st.sidebar:
    st.audio("https://lambda.vgmtreasurechest.com/soundtracks/myst-original-soundtrack/eaouhcni/01.%20Myst%20Theme.mp3", format="audio/mpeg", loop=True, autoplay=True)

# --- Displaying Chat History ---
# Initialise chat history if not already existing
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- CHATBOT ---
# React to user input
if user_message := st.chat_input("Ask about a location or puzzle."):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_message)
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # -- BOT RESPONSE --
    with st.chat_message("assistant"):

        # Have the spinner only for the retrieval
        with st.spinner("Please wait while the bot searches through the documents..."):

            # .stream_chat() make it print bit by bit instead of waiting for the full block
            streaming_response = bot.stream_chat(user_message)

        response = st.write_stream(streaming_response.response_gen)

    # Add full bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})