import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents, read_all_documents_eager
from findex.tokenizer import tokenize


@dataclass(frozen=True)
class CorpusStats:
    mode: str
    docs: int
    tokens: int
    vocab_size: int
    peak_ram_mb: float
    time_sec: float
    top_terms: list[tuple[str, int]]


def compute_stats_lazy(corpus_dir: str | Path, top_n: int = 20) -> CorpusStats:
    """Збір статистики корпусу за один прохід по лінивому конвеєру."""
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count = 0
    token_count = 0
    vocab: Counter[str] = Counter()

    for _, text in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(text):
            token_count += 1
            vocab[token] += 1

    elapsed = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="Lazy (Iterators)",
        docs=doc_count,
        tokens=token_count,
        vocab_size=len(vocab),
        peak_ram_mb=peak_memory / (1024 * 1024),
        time_sec=elapsed,
        top_terms=vocab.most_common(top_n),
    )


def compute_stats_eager(corpus_dir: str | Path, top_n: int = 20) -> CorpusStats:
    """Жадібний збір статистики через матеріалізацію списків у RAM."""
    tracemalloc.start()
    t0 = time.perf_counter()

    all_docs = read_all_documents_eager(corpus_dir)
    all_tokens: list[str] = []
    for _, text in all_docs:
        all_tokens.extend(tokenize(text))

    vocab: Counter[str] = Counter(all_tokens)
    elapsed = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return CorpusStats(
        mode="Eager (Lists)",
        docs=len(all_docs),
        tokens=len(all_tokens),
        vocab_size=len(vocab),
        peak_ram_mb=peak_memory / (1024 * 1024),
        time_sec=elapsed,
        top_terms=vocab.most_common(top_n),
    )
