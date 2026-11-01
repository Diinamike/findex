from typing import List, Set
from findex.indexer import InvertedIndex


def merge_and(postings_a: List[int], postings_b: List[int]) -> List[int]:
    """Двовказівниковий лінійний перетин двох відсортованих списків."""
    result = []
    i, j = 0, 0
    len_a, len_b = len(postings_a), len(postings_b)
    while i < len_a and j < len_b:
        doc_a = postings_a[i]
        doc_b = postings_b[j]
        if doc_a == doc_b:
            result.append(doc_a)
            i += 1
            j += 1
        elif doc_a < doc_b:
            i += 1
        else:
            j += 1
    return result


def merge_or(postings_a: List[int], postings_b: List[int]) -> List[int]:
    """Двовказівникове лінійне об'єднання двох відсортованих списків."""
    result = []
    i, j = 0, 0
    len_a, len_b = len(postings_a), len(postings_b)
    while i < len_a and j < len_b:
        doc_a = postings_a[i]
        doc_b = postings_b[j]
        if doc_a == doc_b:
            result.append(doc_a)
            i += 1
            j += 1
        elif doc_a < doc_b:
            result.append(doc_a)
            i += 1
        else:
            result.append(doc_b)
            j += 1
    while i < len_a:
        result.append(postings_a[i])
        i += 1
    while j < len_b:
        result.append(postings_b[j])
        j += 1
    return result


def merge_not(all_docs: List[int], postings: List[int]) -> List[int]:
    """Різниця множин двовказівниковим методом."""
    result = []
    i, j = 0, 0
    len_all, len_p = len(all_docs), len(postings)
    while i < len_all and j < len_p:
        if all_docs[i] == postings[j]:
            i += 1
            j += 1
        elif all_docs[i] < postings[j]:
            result.append(all_docs[i])
            i += 1
        else:
            j += 1
    while i < len_all:
        result.append(all_docs[i])
        i += 1
    return result


class BooleanSearchEngine:
    def __init__(self, index: InvertedIndex, engine: str = "merge"):
        self.index = index
        self.engine = engine
        self._all_docs = list(range(index.total_docs))

    def _get_doc_ids(self, term: str) -> List[int]:
        postings = self.index.postings.get(term.lower(), [])
        return [p.doc_id for p in postings]

    def search(self, query: str) -> List[int]:
        tokens = query.strip().split()
        if not tokens:
            return []

        if self.engine == "merge":
            return self._search_merge(tokens)
        else:
            return self._search_set(tokens)

    def _search_merge(self, tokens: List[str]) -> List[int]:
        current_docs = None
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.upper() == "OR":
                i += 1
                if i < len(tokens):
                    rhs = self._get_doc_ids(tokens[i])
                    current_docs = merge_or(current_docs or [], rhs)
            elif token.upper() == "NOT":
                i += 1
                if i < len(tokens):
                    rhs = self._get_doc_ids(tokens[i])
                    if current_docs is None:
                        current_docs = merge_not(self._all_docs, rhs)
                    else:
                        not_rhs = merge_not(self._all_docs, rhs)
                        current_docs = merge_and(current_docs, not_rhs)
            else:
                rhs = self._get_doc_ids(token)
                if current_docs is None:
                    current_docs = rhs
                else:
                    current_docs = merge_and(current_docs, rhs)
            i += 1
        return current_docs or []

    def _search_set(self, tokens: List[str]) -> List[int]:
        current_set: Set[int] = None
        all_docs_set = set(self._all_docs)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.upper() == "OR":
                i += 1
                if i < len(tokens):
                    rhs = set(self._get_doc_ids(tokens[i]))
                    current_set = (current_set or set()) | rhs
            elif token.upper() == "NOT":
                i += 1
                if i < len(tokens):
                    rhs = set(self._get_doc_ids(tokens[i]))
                    if current_set is None:
                        current_set = all_docs_set - rhs
                    else:
                        current_set = current_set - rhs
            else:
                rhs = set(self._get_doc_ids(token))
                if current_set is None:
                    current_set = rhs
                else:
                    current_set = current_set & rhs
            i += 1
        return sorted(list(current_set or set()))
