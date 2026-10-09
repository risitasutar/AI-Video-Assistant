import os
import uuid
import threading
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.model_loading import MODEL_LOAD_LOCK

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)  # load the embedding model once per process, not on every analysis
def _load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def get_embeddings():
    # Shared model lock: the background warm-up, a user's analysis and Whisper never load concurrently
    with MODEL_LOAD_LOCK:
        return _load_embeddings()

def warm_up_embeddings():
    """Start loading the embedding model in the background (it takes 10-20 s on a cold start)."""
    threading.Thread(target=get_embeddings, daemon=True).start()

def build_vector_store(transcript : str)->Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    # Session-scoped, in-memory collection with a unique name: a fixed persisted
    # collection would accumulate chunks from every previous video (and, on a shared
    # Streamlit Cloud server, from other users' sessions) into this chat's answers.
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=f"{COLLECTION_NAME}_{uuid.uuid4().hex[:8]}",
    )

    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )


