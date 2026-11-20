from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from findex.index import Index
from findex.tokenizer import tokenize


class QueryNode:
    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def evaluate(self, index: Index) -> Set[int]:
        raise NotImplementedError

    def terms(self) -> List[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> Set[int]:
        if self.value not in index:
            return set()
        return {doc_id for doc_id, _, _ in index[self.value]}

    def terms(self) -> List[str]:
        return [self.value]


@dataclass(frozen=True)
class Phrase(QueryNode):
    words: Tuple[str, ...]

    def __init__(self, words: List[str] | Tuple[str, ...]) -> None:
        object.__setattr__(self, "words", tuple(words))

    def evaluate(self, index: Index) -> Set[int]:
        if not self.words:
            return set()
        if len(self.words) == 1:
            return Term(self.words[0]).evaluate(index)
        if not index.has_positions:
            raise ValueError("Phrase search requires an index built with store_positions=True")

        postings_per_word = []
        for w in self.words:
            if w not in index:
                return set()
            postings_per_word.append({doc_id: set(pos) for doc_id, _, pos in index[w]})

        common_docs = set.intersection(*(set(d.keys()) for d in postings_per_word))
        matched_docs: Set[int] = set()

        for doc_id in common_docs:
            first_positions = postings_per_word[0][doc_id]
            for start_pos in first_positions:
                if all((start_pos + offset) in postings_per_word[offset][doc_id] for offset in range(1, len(self.words))):
                    matched_docs.add(doc_id)
                    break

        return matched_docs

    def terms(self) -> List[str]:
        return list(self.words)


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> Set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> List[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> Set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> List[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> Set[int]:
        all_docs = set(range(index.num_docs))
        return all_docs - self.operand.evaluate(index)

    def terms(self) -> List[str]:
        return []

_TOKEN_RE = re.compile(r'\s*(?:(\()|(\))|("[^"]*")|([^\s()"]+))')


@dataclass(frozen=True)
class _LexToken:
    kind: str
    value: str


def _lex(query: str) -> List[_LexToken]:
    tokens: List[_LexToken] = []
    pos = 0
    while pos < len(query):
        match = _TOKEN_RE.match(query, pos)
        if not match:
            break
        pos = match.end()
        lparen, rparen, phrase, word = match.groups()
        if lparen:
            tokens.append(_LexToken("LPAREN", "("))
        elif rparen:
            tokens.append(_LexToken("RPAREN", ")"))
        elif phrase is not None:
            inner = phrase[1:-1]
            norm_words = list(tokenize(inner))
            if norm_words:
                tokens.append(_LexToken("PHRASE", " ".join(norm_words)))
        elif word is not None:
            upper = word.upper()
            if upper in ("AND", "OR", "NOT"):
                tokens.append(_LexToken(upper, upper))
            else:
                norm_words = list(tokenize(word))
                for w in norm_words:
                    tokens.append(_LexToken("WORD", w))
    return tokens


class QueryParser:
    def __init__(self, tokens: List[_LexToken]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> _LexToken | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> _LexToken:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise ValueError("Empty query")
        node = self._parse_or()
        if self._peek() is not None:
            raise SyntaxError(f"Unexpected token: {self._peek()}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self._peek()) is not None and tok.kind == "OR":
            self._consume()
            right = self._parse_and()
            node = Or(node, right)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self._peek()) is not None and tok.kind in ("AND", "NOT", "LPAREN", "PHRASE", "WORD"):
            if tok.kind == "AND":
                self._consume()
            right = self._parse_not()
            node = And(node, right)
        return node

    def _parse_not(self) -> QueryNode:
        tok = self._peek()
        if tok is not None and tok.kind == "NOT":
            self._consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of query")

        if tok.kind == "LPAREN":
            self._consume()
            node = self._parse_or()
            closing = self._peek()
            if closing is None or closing.kind != "RPAREN":
                raise SyntaxError("Expected closing parenthesis ')'")
            self._consume()
            return node

        if tok.kind == "PHRASE":
            self._consume()
            words = tok.value.split()
            return Term(words[0]) if len(words) == 1 else Phrase(words)

        if tok.kind == "WORD":
            self._consume()
            return Term(tok.value)

        raise SyntaxError(f"Unexpected token in primary expression: {tok}")


def parse_query(query: str) -> QueryNode:
    return QueryParser(_lex(query)).parse()
