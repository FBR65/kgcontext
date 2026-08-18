"""LanceDB vector store — graph-aware retrieval."""

from __future__ import annotations

from typing import Any

from ..types import Document


class SearchResult:
    """A single search result."""

    def __init__(self, document_id: str, score: float, content: str) -> None:
        self.document_id = document_id
        self.score = score
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "content": self.content,
        }


class LanceDBStore:
    """LanceDB-backed vector store. Optional — graph works without it."""

    def __init__(self, uri: str = "./.kgcontext/lancedb") -> None:
        self._uri = uri
        self._db: Any = None
        self._table: Any = None

    def _ensure_db(self) -> None:
        if self._db is None:
            import lancedb

            self._db = lancedb.connect(self._uri)
            try:
                self._table = self._db.open_table("documents")
            except Exception:
                pass

    def index_documents(self, documents: list[Document]) -> None:
        """Embed and store documents in LanceDB."""
        self._ensure_db()
        if not documents:
            return

        # Simple bag-of-words embedding for now (no external embedding model dependency)
        # In production, this would use sentence-transformers or OpenAI embeddings
        import numpy as np

        all_words: set[str] = set()
        for doc in documents:
            all_words.update(doc.content.lower().split())

        vocab = sorted(all_words)
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        data: list[dict[str, Any]] = []
        for doc in documents:
            vec = np.zeros(len(vocab), dtype=np.float32)
            for word in doc.content.lower().split():
                vec[word_to_idx[word]] += 1.0
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            data.append(
                {
                    "id": doc.id,
                    "vector": vec.tolist(),
                    "content": doc.content[:2000],
                    "source_path": doc.source_path,
                }
            )


        if self._table is None:
            self._table = self._db.create_table("documents", data=data)
        else:
            for row in data:
                self._table.add([row])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search for similar documents."""
        self._ensure_db()
        if self._table is None:
            return []

        import numpy as np

        # Build query vector from same vocab
        # Re-derive vocab from stored data
        all_data = list(self._table.to_arrow().to_pylist())
        if not all_data:
            return []

        vocab = sorted(set(word for row in all_data for word in row["content"].lower().split()))
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        query_vec = np.zeros(len(vocab), dtype=np.float32)
        for word in query.lower().split():
            if word in word_to_idx:
                query_vec[word_to_idx[word]] += 1.0
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Cosine similarity with all stored docs
        results: list[SearchResult] = []
        for row in all_data:
            doc_vec = np.array(row["vector"], dtype=np.float32)
            # Re-derive vocab mapping for this doc
            score = float(np.dot(query_vec, doc_vec))
            results.append(
                SearchResult(
                    document_id=row["id"],
                    score=score,
                    content=row["content"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def search_graph_aware(
        self,
        query: str,
        kg: Any,
        k: int = 5,
    ) -> list[SearchResult]:
        """Graph-aware search: incomplete requirements influence query."""
        # Augment query with incomplete requirements
        incomplete = kg.get_incomplete_requirements()
        if incomplete:
            aug = " ".join(r.content for r in incomplete)
            query = f"{query} {aug}"
        return self.search(query, k=k)