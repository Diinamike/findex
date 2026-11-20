from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from findex.index import Index


@dataclass(order=True)
class SearchResult:
    score: float
    doc_id: int
    doc_name: str
    snippet: str = field(default="", compare=False)


@runtime_checkable
class Scorer(Protocol):
    def score_term(self, term: str, tf: int, doc_length: int, index: Index) -> float:
        ...


class TfIdf:
    def score_term(self, term: str, tf: int, doc_length: int, index: Index) -> float:
        df = index.df(term)
        if df == 0 or tf <= 0:
            return 0.0
        tf_weight = 1.0 + math.log(tf)
        idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
        return tf_weight * idf


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score_term(self, term: str, tf: int, doc_length: int, index: Index) -> float:
        df = index.df(term)
        if df == 0 or tf <= 0:
            return 0.0

        n = index.num_docs
        avgdl = index.avg_doc_length or 1.0
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / avgdl))
        tf_norm = (tf * (self.k1 + 1.0)) / denom
        return idf * tf_norm
