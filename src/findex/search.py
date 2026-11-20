from __future__ import annotations

import functools
import heapq
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from findex.index import Index
from findex.query import QueryNode, parse_query
from findex.scoring import BM25, Scorer, SearchResult
from findex.utils import logger, timed


def extract_snippet(text: str, terms: Sequence[str], window: int = 80) -> str:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return ""

    unique_terms = [t for t in dict.fromkeys(terms) if t]
    if not unique_terms:
        return clean_text[: window * 2] + ("..." if len(clean_text) > window * 2 else "")

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in unique_terms) + r")\b",
        re.IGNORECASE | re.UNICODE,
    )

    matches = list(pattern.finditer(clean_text))
    if not matches:
        return clean_text[: window * 2] + ("..." if len(clean_text) > window * 2 else "")

    best_center = matches[0].start()
    best_score = (-1, -1)

    for m in matches:
        c = m.start()
        left = max(0, c - window)
        right = min(len(clean_text), c + window)
        window_matches = [x.group(0).lower() for x in matches if left <= x.start() <= right]
        score = (len(set(window_matches)), len(window_matches))
        if score > best_score:
            best_score = score
            best_center = c

    start = max(0, best_center - window)
    end = min(len(clean_text), best_center + window)

    if start > 0:
        space_idx = clean_text.find(" ", start)
        if 0 < space_idx < best_center:
            start = space_idx + 1
    if end < len(clean_text):
        space_idx = clean_text.rfind(" ", best_center, end)
        if space_idx > best_center:
            end = space_idx

    snippet = clean_text[start:end]
    highlighted = pattern.sub(lambda m: f"[{m.group(0)}]", snippet)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""
    return f"{prefix}{highlighted}{suffix}"


class SearchEngine:
    def __init__(self, index: Index, default_scorer: Optional[Scorer] = None) -> None:
        self.index = index
        self.default_scorer: Scorer = default_scorer or BM25()
        self._cached_match = functools.lru_cache(maxsize=128)(self._match_query_uncached)

    def _match_query_uncached(self, normalized_query: str) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        ast: QueryNode = parse_query(normalized_query)
        matched_ids = tuple(sorted(ast.evaluate(self.index)))
        positive_terms = tuple(dict.fromkeys(ast.terms()))
        return matched_ids, positive_terms

    def match_candidates(self, query: str) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        norm_q = " ".join(query.strip().split())
        before = self._cached_match.cache_info()
        result = self._cached_match(norm_q)
        after = self._cached_match.cache_info()

        status = "CACHE HIT" if after.hits > before.hits else "CACHE MISS"
        logger.info("Query %r -> %s (hits=%d, misses=%d)", norm_q, status, after.hits, after.misses)
        return result

    @property
    def cache_info(self):
        return self._cached_match.cache_info()

    @timed
    def search(
        self,
        query: str,
        top_k: int = 5,
        scorer: Optional[Scorer] = None,
        with_snippets: bool = True,
    ) -> List[SearchResult]:
        active_scorer = scorer or self.default_scorer
        matched_ids, positive_terms = self.match_candidates(query)
        if not matched_ids:
            return []

        candidate_set = set(matched_ids)
        scores: Dict[int, float] = {doc_id: 0.0 for doc_id in candidate_set}

        for term in positive_terms:
            if term not in self.index:
                continue
            for doc_id, tf, _ in self.index[term]:
                if doc_id in candidate_set:
                    doc_len = self.index.get_doc(doc_id).length
                    scores[doc_id] += active_scorer.score_term(term, tf, doc_len, self.index)

        top_pairs = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))

        results: List[SearchResult] = []
        for doc_id, score in top_pairs:
            meta = self.index.get_doc(doc_id)
            snippet = ""
            if with_snippets:
                text = self.index.read_doc_text(doc_id)
                snippet = extract_snippet(text, positive_terms, window=80)
            results.append(
                SearchResult(score=score, doc_id=doc_id, doc_name=meta.name, snippet=snippet)
            )
        return results
