from __future__ import annotations

import json
from typing import Any

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class ChromaDBRetriever:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.embedding_model = SentenceTransformer(config["EMBEDDING_MODEL"])
        self.client = chromadb.PersistentClient(
            path=config["CHROMA_DB_PERSISTENT_DIRECTORY"],
        )
        self.collection = self.client.get_or_create_collection(
            name=config["CHROMA_DB_COLLECTION_NAME"],
        )

    def get_embedding(self, text: str) -> list[float]:
        return self.embedding_model.encode(text).tolist()

    def clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                cleaned[key] = ",".join(map(str, value))
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned

    def index_json(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents, metadatas, ids, embeddings = [], [], [], []
        for item in data:
            text = item["text"]
            chunk_id = f"chunk_{int(item['chunk_id']):04d}"
            metadata = item.get("metadata", {}).copy()
            metadata.pop("text", None)
            metadata.update(
                {
                    "chunk_id": chunk_id,
                    "product": item.get("product"),
                    "plan": item.get("plan"),
                    "content_type": item.get("content_type"),
                    "token_count": item.get("token_count"),
                    "file_name": item.get("file_name"),
                }
            )
            if isinstance(metadata.get("topic_tags"), list):
                metadata["topic_tags"] = ",".join(metadata["topic_tags"])

            documents.append(text)
            metadatas.append(self.clean_metadata(metadata))
            ids.append(chunk_id)
            embeddings.append(self.get_embedding(text))

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def get_count(self) -> int:
        return self.collection.count()

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 5, 50),
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        candidate_docs = []
        vector_scores = {}

        for doc, meta, dist in zip(docs, metas, distances):
            doc_obj = Document(page_content=doc, metadata=meta)
            doc_id = meta["chunk_id"]
            vector_scores[doc_id] = 1 / (1 + dist)
            candidate_docs.append(doc_obj)

        if not candidate_docs:
            return []

        tokenized_query = query.lower().split()
        corpus = [doc.page_content.lower().split() for doc in candidate_docs]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        final_results = []
        for i, doc in enumerate(candidate_docs):
            doc_id = doc.metadata["chunk_id"]
            score = alpha * vector_scores.get(doc_id, 0) + (1 - alpha) * bm25_scores[i]
            final_results.append((doc, float(score)))

        final_results.sort(key=lambda item: item[1], reverse=True)
        return final_results[:k]
