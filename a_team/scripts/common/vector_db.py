
import os
import time
import torch
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseIndexParams

# Embeddings
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel


class LegalVectorDB:
    """
    Qdrant Hybrid Search Client (Dense + Sparse)
    - Dense: SentenceTransformer (Qwen/Qwen3-Embedding-0.6B)
    - Sparse: BGE-M3 (BAAI/bge-m3) - Multilingual/Korean support
    """

    def __init__(self,
                 url: str = None,
                 api_key: str = None,
                 host: str = None,
                 port: int = 6333,
                 dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 sparse_model_name: str = "BAAI/bge-m3",
                 embedding_dim: int = 1024):

        # 1. Qdrant Client
        if url and api_key:
            print(f"🌐 Qdrant Cloud Connect: {url[:30]}...")
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif host:
            print(f"🏠 Qdrant Server Connect: {host}:{port}")
            self.client = QdrantClient(host=host, port=port, timeout=60)
        else:
            print("⚠️ No Connection Info, using Memory Mode")
            self.client = QdrantClient(":memory:")

        # 2. Dense Model
        print(f"🧠 Loading Dense Model: {dense_model_name}")
        self.dense_model = SentenceTransformer(
            dense_model_name, trust_remote_code=True)

        # 3. Sparse Model (BGE-M3)
        print(f"🧠 Loading Sparse Model (BGE-M3): {sparse_model_name}...")
        # BGE-M3 can be used for Dense, Sparse, and ColBERT. We use it for Sparse here.
        # use_fp16=True for speed if GPU available, else False.
        use_fp16 = torch.cuda.is_available()
        self.sparse_model = BGEM3FlagModel(
            sparse_model_name, use_fp16=use_fp16)

        self.embedding_dim = embedding_dim

    def create_collection(self, name: str, recreate: bool = False):
        """Create Qdrant Collection with Dense and Sparse config"""
        collections = [
            c.name for c in self.client.get_collections().collections]

        if name in collections:
            if not recreate:
                print(f"✅ Collection '{name}' exists.")
                return
            else:
                print(f"♻️  Recreating Collection '{name}'...")
                self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        print(f"✨ Collection '{name}' created (Dense + Sparse/BGE-M3)")

    def _get_sparse_vector(self, text: str) -> models.SparseVector:
        """Generate Sparse Vector using BGE-M3"""
        # BGE-M3 returns a dict of token_id: weight
        output = self.sparse_model.encode(
            text, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        # output is like {'lexical_weights': {token_id: weight, ...}}
        weights = output['lexical_weights']

        return models.SparseVector(
            indices=list(map(int, weights.keys())),
            values=list(map(float, weights.values()))
        )

    def _get_batch_sparse_vectors(self, texts: List[str]) -> List[models.SparseVector]:
        """Batch Generate Sparse Vectors using BGE-M3"""
        outputs = self.sparse_model.encode(
            texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        # outputs is a list of dicts if input is list? No, BGEM3 encode returns dict with keys 'lexical_weights' which is list of dicts

        batch_weights = outputs['lexical_weights']  # List[Dict[int, float]]

        sparse_vectors = []
        for weights in batch_weights:
            sparse_vectors.append(models.SparseVector(
                indices=list(map(int, weights.keys())),
                values=list(map(float, weights.values()))
            ))
        return sparse_vectors

    def upsert_chunks(self, collection_name: str, chunks: List[Dict[str, Any]], batch_size: int = 12, start_id: int = 0):
        """Upsert chunks with Hybrid Vectors"""
        if not chunks:
            print("❌ No chunks to upload.")
            return

        total = len(chunks)
        print(f"🚀 Upserting {total} chunks (Batch: {batch_size})...")

        total_saved = 0
        current_id = start_id

        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c['text'][:2000] for c in batch]  # Truncate for safety

            # 1. Dense (Batch)
            dense_vectors = self.dense_model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True)

            # 2. Sparse (Batch - BGE-M3)
            sparse_vectors = self._get_batch_sparse_vectors(texts)

            points = []
            for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(batch, dense_vectors, sparse_vectors)):
                payload = chunk['metadata'].copy()
                payload['text'] = chunk['text']

                points.append(PointStruct(
                    id=current_id + idx,
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": sparse_vec
                    },
                    payload=payload
                ))

            self._upsert_with_retry(collection_name, points)

            total_saved += len(batch)
            current_id += len(batch)
            print(
                f"\r📥 Saved: {total_saved}/{total} ({total_saved/total*100:.1f}%)", end='', flush=True)

        print(f"\n✅ Upload to '{collection_name}' complete!")

    def _upsert_with_retry(self, collection_name, points, max_retries=3):
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=collection_name, points=points)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    print(f"\n❌ Upsert failed: {e}")
                    raise e
