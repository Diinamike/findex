from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from findex.index import Index

ScorerName = Literal["bm25", "tfidf"]
Float64Array = npt.NDArray[np.float64]
Int32Array = npt.NDArray[np.int32]


@dataclass(order=True)
class SearchResult:
    score: float
    doc_id: int
    doc_name: str
    snippet: str = field(default="", compare=False)


@runtime_checkable
class Scorer(Protocol):
    def score_term(self, term: str, tf: int, doc_length: int, index: Index) -> float: ...


class TfIdf:
    def score_term(self, term: str, tf: int, doc_length: int, index: Index) -> float:
        df = index.df(term)
        if df == 0 or tf <= 0:
            return 0.0
        tf_weight = 1.0 + math.log(tf)
        idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
        return tf_weight * idf

    def score_array(
        self,
        tfs: Int32Array,
        doc_lengths: Int32Array,
        df: int,
        num_docs: int,
        avg_doc_length: float,
    ) -> Float64Array:
        if df == 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)
        tf_weight = 1.0 + np.log(tfs.astype(np.float64))
        idf = math.log((num_docs + 1) / (df + 1)) + 1.0
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

    def score_array(
        self,
        tfs: Int32Array,
        doc_lengths: Int32Array,
        df: int,
        num_docs: int,
        avg_doc_length: float,
    ) -> Float64Array:
        if df == 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)

        avgdl = avg_doc_length or 1.0
        idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        tf_f = tfs.astype(np.float64)
        dl_f = doc_lengths.astype(np.float64)
        denom = tf_f + self.k1 * (1.0 - self.b + self.b * (dl_f / avgdl))
        return idf * ((tf_f * (self.k1 + 1.0)) / denom)


def get_scorer(name: ScorerName) -> Scorer:
    if name == "bm25":
        return BM25()
    return TfIdf()
