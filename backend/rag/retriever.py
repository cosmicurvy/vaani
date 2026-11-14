from __future__ import annotations

import pickle

from langchain_classic.retrievers import (ContextualCompressionRetriever,EnsembleRetriever)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from backend.core.config import settings

# Configuration
QDRANT_PATH = settings.QDRANT_PATH
BM25_PATH = settings.BM25_PATH

COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
RERANKER_MODEL = settings.RERANKER_MODEL
EMBEDDING_DEVICE = settings.EMBEDDING_DEVICE

DENSE_K = 10
BM25_K = 10
RERANK_TOP_N = 8
FINAL_TOP_K = 3

class VaaniRetriever:
    def __init__(self) -> None:
        if not QDRANT_PATH.exists():
            raise FileNotFoundError(
                f"Qdrant database not found: {QDRANT_PATH}. "
                "Run backend/rag/indexer.py first.")

        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 documents not found: {BM25_PATH}. "
                "Run backend/rag/indexer.py first.")

        print(f"Loading embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True})

        # Qdrant dense retriever
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            path=str(QDRANT_PATH))
        
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

        # BM25 sparse retriever
        with BM25_PATH.open("rb") as file:
            chunks = pickle.load(file)

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("BM25 document file is empty or invalid. Rebuild the indexes.")

        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = BM25_K

        # Hybrid retrieval (40% BM25 + 60% Dense)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[0.4, 0.6])

        # Cross-encoder reranking
        print(f"Loading reranker: {RERANKER_MODEL} on {EMBEDDING_DEVICE}")
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE})
        
        self.compressor = CrossEncoderReranker(
            model=self.reranker_model,
            top_n=RERANK_TOP_N)

        self.pipeline = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.ensemble_retriever)

        print("Vaani retriever initialized.")

    @staticmethod
    def _deduplicate_faqs(documents: list[Document]) -> list[Document]:
        """Keep only the highest-ranked language variant of each FAQ."""
        results: list[Document] = []
        seen_faq_ids: set[str] = set()

        for doc in documents:
            metadata = doc.metadata or {}

            if metadata.get("source_type") == "faq":
                faq_id = metadata.get("faq_id")
                if faq_id and faq_id in seen_faq_ids:
                    continue
                if faq_id:
                    seen_faq_ids.add(faq_id)

            results.append(doc)
            if len(results) >= FINAL_TOP_K:
                break

        return results

    def search(self, query: str) -> list[Document]:
        """Return the best retrieved documents for a user query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        candidates = self.pipeline.invoke(query.strip())
        return self._deduplicate_faqs(candidates)


if __name__ == "__main__":
    retriever = VaaniRetriever()