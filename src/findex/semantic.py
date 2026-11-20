from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from findex.index import Index

SearchMode = Literal["lexical", "semantic", "hybrid"]
Float32Array = npt.NDArray[np.float32]
Int32Array = npt.NDArray[np.int32]

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class TextEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> Float32Array: ...


def l2_normalize(vectors: Float32Array) -> Float32Array:
    arr: Float32Array = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.sqrt(float((arr * arr).sum())))
        return arr if norm == 0.0 else (arr / norm).astype(np.float32)
    norms = np.sqrt((arr * arr).sum(axis=1, keepdims=True))
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32)


class FastEmbedder:
    """Обгортка над sentence-transformers/all-MiniLM-L6-v2 через ONNX Runtime (fastembed)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> object:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> Float32Array:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = self._get_model()
        raw_vecs: list[Float32Array] = [
            np.asarray(v, dtype=np.float32) for v in model.embed(texts)  # type: ignore[attr-defined]
        ]
        dim = int(raw_vecs[0].shape[0])
        mat = np.empty((len(raw_vecs), dim), dtype=np.float32)
        for i, vec in enumerate(raw_vecs):
            mat[i] = vec
        return l2_normalize(mat)


@dataclass(frozen=True)
class TextChunk:
    doc_id: int
    text: str
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    doc_id: int = 0,
    chunk_words: int = 160,
    overlap_words: int = 32,
) -> list[TextChunk]:
    """Розбиває документ на фрагменти слів із перекриттям, зберігаючи межі символів."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not spans:
        clean = text.strip()
        if not clean:
            return []
        return [TextChunk(doc_id=doc_id, text=clean, char_start=0, char_end=len(text))]

    step = max(1, chunk_words - overlap_words)
    chunks: list[TextChunk] = []
    for start_w in range(0, len(spans), step):
        end_w = min(len(spans), start_w + chunk_words)
        c_start = spans[start_w][0]
        c_end = spans[end_w - 1][1]
        snippet_text = text[c_start:c_end].strip()
        if snippet_text:
            chunks.append(
                TextChunk(doc_id=doc_id, text=snippet_text, char_start=c_start, char_end=c_end)
            )
        if end_w >= len(spans):
            break
    return chunks


def meta_path_for(npy_path: str | Path) -> Path:
    p = Path(npy_path)
    return p.with_suffix(".meta.npz")


class SemanticIndex:
    """Матриця L2-нормалізованих ембедингів фрагментів (.npy) + прив'язка до документів."""

    def __init__(
        self,
        embeddings: Float32Array,
        chunk_doc_ids: Int32Array,
        chunk_starts: Int32Array,
        chunk_ends: Int32Array,
        num_docs: int,
        model_name: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.embeddings: Float32Array = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.chunk_doc_ids: Int32Array = np.asarray(chunk_doc_ids, dtype=np.int32)
        self.chunk_starts: Int32Array = np.asarray(chunk_starts, dtype=np.int32)
        self.chunk_ends: Int32Array = np.asarray(chunk_ends, dtype=np.int32)
        self.num_docs: int = num_docs
        self.model_name: str = model_name

    @property
    def num_chunks(self) -> int:
        return int(self.embeddings.shape[0]) if self.embeddings.ndim == 2 else 0

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    @classmethod
    def build(
        cls,
        index: Index,
        embedder: TextEmbedder | None = None,
        chunk_words: int = 160,
        overlap_words: int = 32,
        batch_size: int = 64,
    ) -> SemanticIndex:
        active_embedder: TextEmbedder = embedder if embedder is not None else FastEmbedder()
        all_chunks: list[TextChunk] = []

        for doc_id in range(index.num_docs):
            meta = index.get_doc(doc_id)
            text = index.read_doc_text(doc_id) if Path(meta.path).exists() else meta.name
            doc_chunks = chunk_text(
                text, doc_id=doc_id, chunk_words=chunk_words, overlap_words=overlap_words
            )
            if not doc_chunks:
                doc_chunks = [
                    TextChunk(doc_id=doc_id, text=meta.name, char_start=0, char_end=len(meta.name))
                ]
            all_chunks.extend(doc_chunks)

        if not all_chunks:
            return cls(
                embeddings=np.empty((0, 0), dtype=np.float32),
                chunk_doc_ids=np.empty(0, dtype=np.int32),
                chunk_starts=np.empty(0, dtype=np.int32),
                chunk_ends=np.empty(0, dtype=np.int32),
                num_docs=index.num_docs,
                model_name=active_embedder.model_name,
            )

        batches: list[Float32Array] = []
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = [c.text for c in all_chunks[i : i + batch_size]]
            batches.append(active_embedder.embed_texts(batch_texts))

        dim = int(batches[0].shape[1])
        matrix = np.empty((len(all_chunks), dim), dtype=np.float32)
        offset = 0
        for batch in batches:
            n_rows = int(batch.shape[0])
            matrix[offset : offset + n_rows] = batch
            offset += n_rows

        doc_ids = np.array([c.doc_id for c in all_chunks], dtype=np.int32)
        starts = np.array([c.char_start for c in all_chunks], dtype=np.int32)
        ends = np.array([c.char_end for c in all_chunks], dtype=np.int32)

        return cls(
            embeddings=matrix,
            chunk_doc_ids=doc_ids,
            chunk_starts=starts,
            chunk_ends=ends,
            num_docs=index.num_docs,
            model_name=active_embedder.model_name,
        )

    def save(self, npy_path: str | Path) -> None:
        out_npy = Path(npy_path)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, self.embeddings)  # type: ignore[reportUnknownMemberType]
        meta_file = meta_path_for(out_npy)
        np.savez_compressed(  # type: ignore[reportUnknownMemberType]
            meta_file,
            chunk_doc_ids=self.chunk_doc_ids,
            chunk_starts=self.chunk_starts,
            chunk_ends=self.chunk_ends,
            meta_json=np.array(
                json.dumps({"num_docs": self.num_docs, "model_name": self.model_name})
            ),
        )

    @classmethod
    def load(cls, npy_path: str | Path) -> SemanticIndex:
        in_npy = Path(npy_path)
        if not in_npy.exists():
            raise FileNotFoundError(f"Embeddings matrix not found: {in_npy}")
        meta_file = meta_path_for(in_npy)
        if not meta_file.exists():
            raise FileNotFoundError(f"Embeddings metadata not found: {meta_file}")

        matrix = np.asarray(np.load(in_npy), dtype=np.float32)  # type: ignore[reportUnknownMemberType]
        with np.load(meta_file) as data:  # type: ignore[reportUnknownMemberType]
            doc_ids = np.asarray(data["chunk_doc_ids"], dtype=np.int32)
            starts = np.asarray(data["chunk_starts"], dtype=np.int32)
            ends = np.asarray(data["chunk_ends"], dtype=np.int32)
            meta_info = json.loads(str(data["meta_json"].item()))

        return cls(
            embeddings=matrix,
            chunk_doc_ids=doc_ids,
            chunk_starts=starts,
            chunk_ends=ends,
            num_docs=int(meta_info["num_docs"]),
            model_name=str(meta_info.get("model_name", DEFAULT_EMBED_MODEL)),
        )

    def score_documents(
        self, query_vec: Float32Array
    ) -> tuple[npt.NDArray[np.float32], Int32Array]:
        """Множить матрицю чанків на вектор запиту та повертає (max_doc_scores, best_chunk_idx)."""
        if self.num_chunks == 0 or self.num_docs == 0:
            return (
                np.full(self.num_docs, -np.inf, dtype=np.float32),
                np.full(self.num_docs, -1, dtype=np.int32),
            )

        q_norm = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        chunk_sims: Float32Array = self.embeddings @ q_norm

        best_doc_scores = np.full(self.num_docs, -np.inf, dtype=np.float32)
        best_chunk_idx = np.full(self.num_docs, -1, dtype=np.int32)

        for c_idx, (doc_id, sim) in enumerate(
            zip(self.chunk_doc_ids.tolist(), chunk_sims.tolist(), strict=True)
        ):
            if 0 <= doc_id < self.num_docs and sim > float(best_doc_scores[doc_id]):
                best_doc_scores[doc_id] = sim
                best_chunk_idx[doc_id] = c_idx

        return best_doc_scores, best_chunk_idx


def reciprocal_rank_fusion(
    lexical_doc_ids: list[int],
    semantic_doc_ids: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Об'єднує два відсортовані списки doc_id за формулою RRF: sum(1 / (k + rank))."""
    rrf_scores: dict[int, float] = {}
    for rank, doc_id in enumerate(lexical_doc_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(semantic_doc_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda pair: (-pair[1], -pair[0]))
