from __future__ import annotations

import functools
import heapq
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from findex.index import Index
from findex.query import EngineName, QuerySyntaxError, extract_positive_terms, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.semantic import (
    FastEmbedder,
    SearchMode,
    SemanticIndex,
    TextEmbedder,
    reciprocal_rank_fusion,
)
from findex.utils import timed

logger = logging.getLogger("findex")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


def build_snippet(text: str, terms: list[str], window: int = 80) -> str:
    if not text:
        return ""
    if not terms:
        clean = " ".join(text[: window * 2].split())
        return clean + ("..." if len(text) > window * 2 else "")

    unique_terms = list(dict.fromkeys(terms))
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in unique_terms) + r")\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        clean = " ".join(text[: window * 2].split())
        return clean + ("..." if len(text) > window * 2 else "")

    best_center = matches[0].start()
    best_score = (-1, -1)

    for i, m in enumerate(matches):
        center = m.start()
        seen: set[str] = set()
        total_hits = 0
        for j in range(i, len(matches)):
            if matches[j].start() - center > window * 2:
                break
            seen.add(matches[j].group(0).lower())
            total_hits += 1
        cand_score = (len(seen), total_hits)
        if cand_score > best_score:
            best_score = cand_score
            best_center = center

    start = max(0, best_center - window)
    end = min(len(text), best_center + window)
    raw_slice = text[start:end]
    highlighted = pattern.sub(lambda m: f"[{m.group(0)}]", raw_slice)
    snippet = " ".join(highlighted.split())

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _extract_plain_terms(raw_query: str) -> list[str]:
    clean = re.sub(r'\b(AND|OR|NOT)\b|[()"]', " ", raw_query)
    return [tok.lower() for tok in re.findall(r"\w+", clean) if len(tok) >= 2]


class SearchEngine:
    def __init__(
        self,
        index: Index,
        default_scorer: Scorer | None = None,
        cache_size: int = 128,
        semantic_index: SemanticIndex | None = None,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self.index = index
        self.default_scorer: Scorer = default_scorer if default_scorer is not None else BM25()
        self.semantic_index: SemanticIndex | None = semantic_index
        self._embedder: TextEmbedder | None = embedder
        self._cached_candidates = functools.lru_cache(maxsize=cache_size)(
            self._resolve_candidates_uncached
        )

    @property
    def has_semantic(self) -> bool:
        return self.semantic_index is not None and self.semantic_index.num_chunks > 0

    def _get_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            model_name = (
                self.semantic_index.model_name
                if self.semantic_index is not None
                else "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._embedder = FastEmbedder(model_name=model_name)
        return self._embedder

    def _resolve_candidates_uncached(
        self, normalized_query: str, engine: EngineName
    ) -> tuple[int, ...]:
        ast = parse_query(normalized_query)
        matched = ast.evaluate(self.index, engine=engine)
        return tuple(sorted(matched))

    def candidate_doc_ids(self, raw_query: str, engine: EngineName = "set") -> tuple[int, ...]:
        normalized = " ".join(raw_query.strip().split())
        before = self._cached_candidates.cache_info()
        result = self._cached_candidates(normalized, engine)
        after = self._cached_candidates.cache_info()

        if after.hits > before.hits:
            logger.info(
                "Query %r -> CACHE HIT (hits=%d, misses=%d)",
                normalized,
                after.hits,
                after.misses,
            )
        else:
            logger.info(
                "Query %r -> CACHE MISS (hits=%d, misses=%d)",
                normalized,
                after.hits,
                after.misses,
            )
        return result

    @property
    def cache_info(self) -> CacheStats:
        info = self._cached_candidates.cache_info()
        return CacheStats(hits=info.hits, misses=info.misses, maxsize=info.maxsize, currsize=info.currsize)

    def _score_lexical_python(
        self,
        candidates: tuple[int, ...],
        positive_terms: list[str],
        active_scorer: Scorer,
        top_k: int,
    ) -> list[SearchResult]:
        """Референсний пооб'єктний скорер Лаби 3."""
        scores: dict[int, float] = {doc_id: 0.0 for doc_id in candidates}
        for term in positive_terms:
            if term not in self.index:
                continue
            for doc_id, tf, _ in self.index[term]:
                if doc_id in scores:
                    doc_len = self.index.get_doc(doc_id).length
                    scores[doc_id] += active_scorer.score_term(term, tf, doc_len, self.index)

        ranked_raw = [
            SearchResult(
                score=score,
                doc_id=doc_id,
                doc_name=self.index.get_doc(doc_id).name,
            )
            for doc_id, score in scores.items()
        ]
        return heapq.nlargest(top_k, ranked_raw)

    def _score_lexical_numpy(
        self,
        candidates: tuple[int, ...],
        positive_terms: list[str],
        active_scorer: BM25 | TfIdf,
        top_k: int,
    ) -> list[SearchResult]:
        """Векторизований скоринг над масивами NumPy int32 + відбір топ-k через np.argpartition."""
        n_docs = self.index.num_docs
        if n_docs == 0 or top_k <= 0 or not candidates:
            return []

        doc_lengths = self.index.doc_lengths
        avgdl = self.index.avg_doc_length
        total_scores = np.zeros(n_docs, dtype=np.float64)

        for term in positive_terms:
            d_ids, tfs = self.index.postings_arrays(term)
            if d_ids.size == 0:
                continue
            term_weights = active_scorer.score_array(
                tfs=tfs,
                doc_lengths=doc_lengths[d_ids],
                df=int(d_ids.size),
                num_docs=n_docs,
                avg_doc_length=avgdl,
            )
            total_scores[d_ids] += term_weights

        cand_arr = np.asarray(candidates, dtype=np.int32)
        cand_scores = total_scores[cand_arr]
        m = cand_arr.size

        if m <= top_k:
            shortlist_idx = np.arange(m)
        else:
            kth_idx = m - top_k
            part_idx = cand_scores.argpartition(kth_idx)
            threshold = float(cand_scores[part_idx[kth_idx]])
            shortlist_idx = (cand_scores >= threshold).nonzero()[0]

        ranked_shortlist = [
            SearchResult(
                score=float(cand_scores[idx_i]),
                doc_id=int(cand_arr[idx_i]),
                doc_name=self.index.get_doc(int(cand_arr[idx_i])).name,
            )
            for idx_i in shortlist_idx.tolist()
        ]
        return heapq.nlargest(top_k, ranked_shortlist)

    def _run_lexical_ranking(
        self,
        raw_query: str,
        top_k: int,
        active_scorer: Scorer,
        engine: EngineName,
    ) -> tuple[list[SearchResult], list[str]]:
        ast = parse_query(raw_query)
        positive_terms = extract_positive_terms(ast)
        candidates = self.candidate_doc_ids(raw_query, engine=engine)
        if not candidates or top_k <= 0:
            return [], positive_terms

        if (
            isinstance(active_scorer, (BM25, TfIdf))
            and hasattr(self.index, "postings_arrays")
            and hasattr(self.index, "doc_lengths")
        ):
            ranked = self._score_lexical_numpy(candidates, positive_terms, active_scorer, top_k)
        else:
            ranked = self._score_lexical_python(candidates, positive_terms, active_scorer, top_k)
        return ranked, positive_terms

    def _run_semantic_ranking(
        self,
        raw_query: str,
        top_k: int,
    ) -> tuple[list[SearchResult], dict[int, int]]:
        if self.semantic_index is None or self.semantic_index.num_chunks == 0 or top_k <= 0:
            return [], {}

        clean_q = re.sub(r'\b(AND|OR|NOT)\b|[()"]', " ", raw_query).strip()
        if not clean_q:
            return [], {}

        embedder = self._get_embedder()
        q_mat = embedder.embed_texts([clean_q])
        if q_mat.size == 0:
            return [], {}

        doc_scores, best_chunks = self.semantic_index.score_documents(q_mat[0])
        valid_mask = np.isfinite(doc_scores)
        valid_ids = valid_mask.nonzero()[0]
        if valid_ids.size == 0:
            return [], {}

        valid_scores = doc_scores[valid_ids]
        ranked_raw = [
            SearchResult(
                score=float(sc),
                doc_id=int(d_id),
                doc_name=self.index.get_doc(int(d_id)).name,
            )
            for d_id, sc in zip(valid_ids.tolist(), valid_scores.tolist(), strict=True)
        ]
        top_sem = heapq.nlargest(top_k, ranked_raw)
        best_chunk_map = {
            item.doc_id: int(best_chunks[item.doc_id])
            for item in top_sem
            if int(best_chunks[item.doc_id]) >= 0
        }
        return top_sem, best_chunk_map

    def _attach_snippets(
        self,
        results: list[SearchResult],
        terms: list[str],
        best_chunk_map: dict[int, int] | None = None,
    ) -> None:
        for item in results:
            meta = self.index.get_doc(item.doc_id)
            if not Path(meta.path).exists():
                continue
            text = self.index.read_doc_text(item.doc_id)
            if (
                best_chunk_map is not None
                and self.semantic_index is not None
                and item.doc_id in best_chunk_map
            ):
                c_idx = best_chunk_map[item.doc_id]
                c_start = int(self.semantic_index.chunk_starts[c_idx])
                c_end = int(self.semantic_index.chunk_ends[c_idx])
                chunk_slice = text[c_start:c_end]
                item.snippet = build_snippet(chunk_slice, terms, window=95)
            else:
                item.snippet = build_snippet(text, terms, window=80)

    @timed
    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        scorer: Scorer | None = None,
        engine: EngineName = "set",
        with_snippets: bool = True,
        mode: SearchMode = "lexical",
    ) -> list[SearchResult]:
        active_scorer = scorer if scorer is not None else self.default_scorer
        if top_k <= 0:
            return []

        if mode == "lexical":
            top_results, positive_terms = self._run_lexical_ranking(
                raw_query, top_k=top_k, active_scorer=active_scorer, engine=engine
            )
            if with_snippets:
                self._attach_snippets(top_results, positive_terms)
            return top_results

        if not self.has_semantic:
            raise ValueError(
                "Semantic embeddings are not loaded. Run `findex embed` or configure EMBEDDINGS_PATH."
            )

        if mode == "semantic":
            top_results, chunk_map = self._run_semantic_ranking(raw_query, top_k=top_k)
            if with_snippets:
                self._attach_snippets(top_results, _extract_plain_terms(raw_query), chunk_map)
            return top_results

        # mode == "hybrid": об'єднуємо лексичний та семантичний рейтинги через RRF
        pool_size = max(top_k * 3, 30)
        try:
            lex_results, positive_terms = self._run_lexical_ranking(
                raw_query, top_k=pool_size, active_scorer=active_scorer, engine=engine
            )
        except QuerySyntaxError:
            lex_results, positive_terms = [], _extract_plain_terms(raw_query)

        sem_results, chunk_map = self._run_semantic_ranking(raw_query, top_k=pool_size)

        fused = reciprocal_rank_fusion(
            lexical_doc_ids=[r.doc_id for r in lex_results],
            semantic_doc_ids=[r.doc_id for r in sem_results],
            k=60,
        )
        top_results = [
            SearchResult(
                score=rrf_score,
                doc_id=doc_id,
                doc_name=self.index.get_doc(doc_id).name,
            )
            for doc_id, rrf_score in fused[:top_k]
        ]
        if with_snippets:
            terms = positive_terms if positive_terms else _extract_plain_terms(raw_query)
            self._attach_snippets(top_results, terms, chunk_map)
        return top_results
