import os
import pickle
import struct
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from findex.corpus import iter_documents
from findex.models import DocMeta, Posting
from findex.tokenizer import tokenize


# Моделі без slots для бенчмарку варіантів пам'яті
@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    term_frequency: int


class InvertedIndex:
    def __init__(self, with_positions: bool = False):
        self.with_positions: bool = with_positions
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_meta: Dict[int, DocMeta] = {}
        self.total_docs: int = 0

    def build_from_directory(self, corpus_path: Path):
        for doc_id, (filename, text) in enumerate(iter_documents(corpus_path)):
            tokens = list(tokenize(text))
            doc_len = len(tokens)
            self.doc_meta[doc_id] = DocMeta(doc_id=doc_id, name=filename, length=doc_len)

            if self.with_positions:
                term_positions = defaultdict(list)
                for pos, t in enumerate(tokens):
                    term_positions[t].append(pos)
                for t, pos_list in term_positions.items():
                    self.postings[t].append(
                        Posting(doc_id=doc_id, term_frequency=len(pos_list), positions=tuple(pos_list))
                    )
            else:
                counts = Counter(tokens)
                for t, freq in counts.items():
                    self.postings[t].append(Posting(doc_id=doc_id, term_frequency=freq))

            self.total_docs += 1

    def save_pickle(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle(cls, path: Path) -> "InvertedIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_binary(self, path: Path):
        """
        Компактний бінарний формат на основі array('I'):
        Header: total_docs (uint32), num_terms (uint32)
        DocMeta table: doc_id, length, filename
        Postings table: term_bytes, doc_ids array, freqs array
        """
        with open(path, "wb") as f:
            f.write(struct.pack("<II", self.total_docs, len(self.postings)))

            # DocMeta
            for doc_id in range(self.total_docs):
                meta = self.doc_meta[doc_id]
                name_bytes = meta.name.encode("utf-8")
                f.write(struct.pack("<IIH", meta.doc_id, meta.length, len(name_bytes)))
                f.write(name_bytes)

            # Postings
            for term, post_list in self.postings.items():
                term_b = term.encode("utf-8")
                n_postings = len(post_list)
                f.write(struct.pack("<HH", len(term_b), n_postings))
                f.write(term_b)

                doc_ids = array("I", (p.doc_id for p in post_list))
                freqs = array("I", (p.term_frequency for p in post_list))
                doc_ids.tofile(f)
                freqs.tofile(f)

    @classmethod
    def load_binary(cls, path: Path) -> "InvertedIndex":
        idx = cls(with_positions=False)
        with open(path, "rb") as f:
            idx.total_docs, num_terms = struct.unpack("<II", f.read(8))

            for _ in range(idx.total_docs):
                doc_id, length, name_len = struct.unpack("<IIH", f.read(10))
                name = f.read(name_len).decode("utf-8")
                idx.doc_meta[doc_id] = DocMeta(doc_id=doc_id, name=name, length=length)

            for _ in range(num_terms):
                term_len, n_postings = struct.unpack("<HH", f.read(4))
                term = f.read(term_len).decode("utf-8")

                doc_ids = array("I")
                doc_ids.fromfile(f, n_postings)
                freqs = array("I")
                freqs.fromfile(f, n_postings)

                idx.postings[term] = [
                    Posting(doc_id=doc_ids[i], term_frequency=freqs[i])
                    for i in range(n_postings)
                ]
        return idx


def benchmark_memory_variants(corpus_path: Path):
    """Вимірює використання RAM під tracemalloc для 3 варіантів зберігання."""
    print("--- Профілювання трьох варіантів зберігання постінгів ---")

    # Варіант 1: Звичайний dataclass (без __slots__)
    tracemalloc.start()
    t0 = time.perf_counter()
    p1: Dict[str, List[PlainPosting]] = defaultdict(list)
    for doc_id, (_, text) in enumerate(iter_documents(corpus_path)):
        for t, freq in Counter(tokenize(text)).items():
            p1[t].append(PlainPosting(doc_id=doc_id, term_frequency=freq))
    t1 = time.perf_counter() - t0
    peak1 = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    del p1

    # Варіант 2: Dataclass зі slots=True
    tracemalloc.start()
    t0 = time.perf_counter()
    p2: Dict[str, List[Posting]] = defaultdict(list)
    for doc_id, (_, text) in enumerate(iter_documents(corpus_path)):
        for t, freq in Counter(tokenize(text)).items():
            p2[t].append(Posting(doc_id=doc_id, term_frequency=freq))
    t2 = time.perf_counter() - t0
    peak2 = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    del p2

    # Варіант 3: Пари array('I')
    tracemalloc.start()
    t0 = time.perf_counter()
    p3_docs: Dict[str, array] = defaultdict(lambda: array("I"))
    p3_freqs: Dict[str, array] = defaultdict(lambda: array("I"))
    for doc_id, (_, text) in enumerate(iter_documents(corpus_path)):
        for t, freq in Counter(tokenize(text)).items():
            p3_docs[t].append(doc_id)
            p3_freqs[t].append(freq)
    t3 = time.perf_counter() - t0
    peak3 = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()
    del p3_docs, p3_freqs

    return [
        ("Plain dataclass (з __dict__)", peak1, t1),
        ("Dataclass (slots=True)", peak2, t2),
        ("Пари array('I')", peak3, t3),
    ]
