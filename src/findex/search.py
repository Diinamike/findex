from __future__ import annotations

import functools
import heapq
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from findex.index import Index
from findex.query import EngineName, extract_positive_terms, parse_query
from findex.scoring import BM25, Scorer, SearchResult
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


class SearchEngine:
    def __init__(
        self,
        index: Index,
        default_scorer: Scorer | None = None,
        cache_size: int = 128,
    ) -> None:
        self.index = index
        self.default_scorer: Scorer = default_scorer if default_scorer is not None else BM25()
        self._cached_candidates = functools.lru_cache(maxsize=cache_size)(
            self._resolve_candidates_uncached
        )

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

    @timed
    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        scorer: Scorer | None = None,
        engine: EngineName = "set",
        with_snippets: bool = True,
    ) -> list[SearchResult]:
        active_scorer = scorer if scorer is not None else self.default_scorer
        ast = parse_query(raw_query)
        positive_terms = extract_positive_terms(ast)
        candidates = set(self.candidate_doc_ids(raw_query, engine=engine))

        if not candidates:
            return []

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
        top_results = heapq.nlargest(top_k, ranked_raw)

        if with_snippets:
            for item in top_results:
                meta = self.index.get_doc(item.doc_id)
                if Path(meta.path).exists():
                    text = self.index.read_doc_text(item.doc_id)
                    item.snippet = build_snippet(text, positive_terms, window=80)

        return top_results
