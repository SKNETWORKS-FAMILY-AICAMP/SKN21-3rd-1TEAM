from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "criminal"


def get_embeddings():
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def get_client():
    return QdrantClient(url=QDRANT_URL)


def get_vector_store():
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(), collection_name=COLLECTION_NAME, url=QDRANT_URL
    )


def init_vector_store(documents):
    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    return QdrantVectorStore.from_documents(
        documents,
        get_embeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
