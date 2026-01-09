
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Load env from the chatbot script directory
_DOTENV_PATH = Path(
    "a_team/scripts/architectures/chatbot_graph_V8_FINAL.py").with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")


def enable_sparse_vectors():
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("Error: QDRANT_URL or QDRANT_API_KEY not found.")
        return

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    print(f"Connecting to Qdrant at {QDRANT_URL}...")

    # 1. 컬렉션 정보 확인
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"Current Config for '{COLLECTION_NAME}':")
        print(f" - Vectors: {collection_info.config.params.vectors}")
        print(
            f" - Sparse Vectors: {collection_info.config.params.sparse_vectors}")

        if collection_info.config.params.sparse_vectors:
            print("\n✅ Sparse vectors are ALREADY enabled. Skipping migration.")
            return

    except Exception as e:
        print(f"Error getting collection info: {e}")
        return

    # 2. Sparse Vector 설정 추가 (BGE-M3 모델 사용 예정이므로 이름은 'sparse'로 통일)
    print("\n🚀 Enabling 'sparse' vector index...")

    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,  # 메모리에 상주하여 속도 최적화
                    )
                )
            }
        )
        print("✅ Migration Successful! Sparse vector 'sparse' is now enabled.")
        print(
            "ℹ️ Note: Existing points do not have sparse data yet. You need to update them.")

    except Exception as e:
        print(f"❌ Migration Failed: {e}")


if __name__ == "__main__":
    enable_sparse_vectors()
