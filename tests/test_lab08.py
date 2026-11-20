from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_parallel_index
from findex.query import extract_positive_terms, parse_query
from findex.scoring import BM25, TfIdf
from findex.search import SearchEngine
from findex.semantic import (
    Float32Array,
    SemanticIndex,
    chunk_text,
    l2_normalize,
    reciprocal_rank_fusion,
)

runner = CliRunner()


class DeterministicStubEmbedder:
    """Швидкий детермінований ембедер (bag-of-words projection) для офлайн-тестів без HuggingFace."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "stub-minilm-32d"

    def embed_texts(self, texts: list[str]) -> Float32Array:
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row_i, text in enumerate(texts):
            for word in text.lower().split():
                clean = "".join(ch for ch in word if ch.isalnum())
                if not clean:
                    continue
                bucket = sum(ord(c) * (idx + 1) for idx, c in enumerate(clean)) % self._dim
                mat[row_i, bucket] += 1.0
        return l2_normalize(mat)


def _create_mini_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = {
        "00_re.txt": "Regular expression pattern matching with re compile search match groups.",
        "01_datetime.txt": "Date and time manipulation with datetime timezone utc offset timedelta.",
        "02_heapq.txt": "Heap queue algorithm priority queue smallest largest heappush heappop.",
        "03_struct.txt": "Pack and unpack binary data buffers into raw bytes for C structs.",
        "04_difflib.txt": "Compare sequences and files line by line generating unified diff.",
        "05_copy.txt": "Shallow and deep copy operations for nested objects without mutation.",
    }
    for name, body in docs.items():
        (corpus / name).write_text(body, encoding="utf-8")
    return corpus


def test_numpy_scorer_matches_lab03_python_reference(tmp_path: Path) -> None:
    corpus = _create_mini_corpus(tmp_path)
    idx, _ = build_parallel_index(corpus, workers=1, executor_kind="serial")
    engine = SearchEngine(idx)

    for scorer in (BM25(), TfIdf()):
        for query in ("pattern OR queue OR binary", "regular AND expression", "copy"):
            ast = parse_query(query)
            terms = extract_positive_terms(ast)
            cands = engine.candidate_doc_ids(query)

            py_res = engine._score_lexical_python(cands, terms, scorer, top_k=3)
            np_res = engine._score_lexical_numpy(cands, terms, scorer, top_k=3)

            assert [(r.doc_id, r.doc_name) for r in py_res] == [
                (r.doc_id, r.doc_name) for r in np_res
            ]
            for r_py, r_np in zip(py_res, np_res, strict=True):
                assert np.isclose(r_py.score, r_np.score, rtol=1e-11, atol=1e-11)


def test_semantic_index_save_load_and_hybrid_rrf(tmp_path: Path) -> None:
    corpus = _create_mini_corpus(tmp_path)
    idx, _ = build_parallel_index(corpus, workers=1, executor_kind="serial")
    embedder = DeterministicStubEmbedder(dim=32)

    sem_idx = SemanticIndex.build(idx, embedder=embedder, chunk_words=64, overlap_words=8)
    npy_path = tmp_path / "prod_embeddings.npy"
    sem_idx.save(npy_path)

    loaded_sem = SemanticIndex.load(npy_path)
    assert loaded_sem.num_docs == idx.num_docs
    assert loaded_sem.num_chunks == sem_idx.num_chunks
    assert loaded_sem.dim == 32

    engine = SearchEngine(idx, semantic_index=loaded_sem, embedder=embedder)

    sem_hits = engine.search("binary raw bytes buffers", top_k=3, mode="semantic")
    assert sem_hits
    assert sem_hits[0].doc_name == "03_struct.txt"
    assert sem_hits[0].snippet != ""

    hyb_hits = engine.search("binary OR buffers", top_k=3, mode="hybrid")
    assert hyb_hits
    assert hyb_hits[0].doc_name == "03_struct.txt"

    fused = reciprocal_rank_fusion([1, 2, 3], [2, 1, 4], k=60)
    assert fused[0][0] == 2
    assert fused[1][0] == 1


def test_chunk_text_overlap() -> None:
    words = [f"word{i}" for i in range(50)]
    chunks = chunk_text(" ".join(words), doc_id=7, chunk_words=20, overlap_words=5)
    assert len(chunks) == 3
    assert all(c.doc_id == 7 for c in chunks)
    assert chunks[0].text.startswith("word0 ")
    assert "word15" in chunks[0].text and "word15" in chunks[1].text


def test_cli_embed_and_hybrid_search(tmp_path: Path, monkeypatch: Any) -> None:
    corpus = _create_mini_corpus(tmp_path)
    idx_path = tmp_path / "index.bin"
    emb_path = tmp_path / "prod_embeddings.npy"

    monkeypatch.setattr("findex.cli.FastEmbedder", lambda model_name="": DeterministicStubEmbedder(16))
    monkeypatch.setattr("findex.search.FastEmbedder", lambda model_name="": DeterministicStubEmbedder(16))

    res_idx = runner.invoke(app, ["index", str(corpus), "--out", str(idx_path), "--executor", "serial"])
    assert res_idx.exit_code == 0

    res_emb = runner.invoke(app, ["embed", str(idx_path), "--out", str(emb_path)])
    assert res_emb.exit_code == 0
    assert emb_path.exists()

    res_srch = runner.invoke(
        app,
        ["search", str(idx_path), "priority queue", "--mode", "hybrid", "--embeddings", str(emb_path), "--json"],
    )
    assert res_srch.exit_code == 0
    assert "02_heapq.txt" in res_srch.stdout


def test_benchmark_numpy_scoring(benchmark: Any, tmp_path: Path) -> None:
    corpus = _create_mini_corpus(tmp_path)
    idx, _ = build_parallel_index(corpus, workers=1, executor_kind="serial")
    engine = SearchEngine(idx)
    scorer = BM25()

    def run_query() -> int:
        hits = engine.search(
            "pattern OR queue OR binary",
            top_k=5,
            scorer=scorer,
            with_snippets=False,
            mode="lexical",
        )
        return len(hits)

    count = benchmark(run_query)
    assert count == 3


def test_benchmark_semantic_matmul(benchmark: Any) -> None:
    rng = np.random.default_rng(42)
    mat = l2_normalize(rng.standard_normal((2000, 384), dtype=np.float32))
    q_vec = l2_normalize(rng.standard_normal(384, dtype=np.float32))

    def matmul_step() -> float:
        scores = mat @ q_vec
        return float(scores.max())

    top_score = benchmark(matmul_step)
    assert -1.0 <= top_score <= 1.0
