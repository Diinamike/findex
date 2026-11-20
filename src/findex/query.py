from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from findex.index import Index, merge_postings
from findex.tokenizer import tokenize

EngineName = Literal["set", "merge"]


class QuerySyntaxError(ValueError):
    """Синтаксична помилка в пошуковому запиті."""


class QueryNode:
    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        raise NotImplementedError

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        return set(index.doc_ids_for(self.value))


@dataclass(frozen=True)
class Phrase(QueryNode):
    terms: tuple[str, ...]

    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        if not self.terms:
            return set()
        if not index.has_positions:
            raise ValueError("Phrase queries require an index built with store_positions=True")
        if len(self.terms) == 1:
            return set(index.doc_ids_for(self.terms[0]))

        term_maps: list[dict[int, set[int]]] = []
        for term in self.terms:
            if term not in index:
                return set()
            term_maps.append({doc_id: set(pos) for doc_id, _, pos in index[term]})

        common_docs = set(term_maps[0].keys())
        for tm in term_maps[1:]:
            common_docs &= tm.keys()

        matched: set[int] = set()
        for doc_id in common_docs:
            start_positions = term_maps[0][doc_id]
            for start in start_positions:
                if all((start + offset) in term_maps[offset][doc_id] for offset in range(1, len(self.terms))):
                    matched.add(doc_id)
                    break
        return matched


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        if engine == "merge" and isinstance(self.left, Term) and isinstance(self.right, Term):
            l_ids = index.doc_ids_for(self.left.value)
            r_ids = index.doc_ids_for(self.right.value)
            return set(merge_postings(l_ids, r_ids))
        left_docs = self.left.evaluate(index, engine=engine)
        if not left_docs:
            return set()
        return left_docs & self.right.evaluate(index, engine=engine)


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        return self.left.evaluate(index, engine=engine) | self.right.evaluate(index, engine=engine)


@dataclass(frozen=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index, engine: EngineName = "set") -> set[int]:
        all_docs = set(range(index.num_docs))
        return all_docs - self.operand.evaluate(index, engine=engine)


@dataclass(frozen=True)
class _Token:
    kind: Literal["LPAREN", "RPAREN", "AND", "OR", "NOT", "PHRASE", "WORD"]
    value: str


def _lex(raw_query: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(raw_query)

    while i < n:
        ch = raw_query[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append(_Token("LPAREN", "("))
            i += 1
        elif ch == ")":
            tokens.append(_Token("RPAREN", ")"))
            i += 1
        elif ch == '"':
            end = raw_query.find('"', i + 1)
            if end == -1:
                raise QuerySyntaxError("Unclosed quote in query")
            phrase_raw = raw_query[i + 1 : end]
            tokens.append(_Token("PHRASE", phrase_raw))
            i = end + 1
        else:
            start = i
            while i < n and not raw_query[i].isspace() and raw_query[i] not in '()"':
                i += 1
            word = raw_query[start:i]
            if word == "AND":
                tokens.append(_Token("AND", word))
            elif word == "OR":
                tokens.append(_Token("OR", word))
            elif word == "NOT":
                tokens.append(_Token("NOT", word))
            else:
                tokens.append(_Token("WORD", word))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Empty query")
        node = self._parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Unexpected token: {self.peek()}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self.peek()) is not None and tok.kind == "OR":
            self.consume()
            node = Or(node, self._parse_and())
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self.peek()) is not None:
            if tok.kind == "AND":
                self.consume()
                node = And(node, self._parse_not())
            elif tok.kind in ("WORD", "PHRASE", "LPAREN", "NOT"):
                node = And(node, self._parse_not())
            else:
                break
        return node

    def _parse_not(self) -> QueryNode:
        tok = self.peek()
        if tok is not None and tok.kind == "NOT":
            self.consume()
            return Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")
        if tok.kind == "LPAREN":
            self.consume()
            node = self._parse_or()
            closing = self.peek()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Unclosed parenthesis '(' in query")
            self.consume()
            return node
        if tok.kind == "PHRASE":
            self.consume()
            words = tuple(tokenize(tok.value))
            if not words:
                raise QuerySyntaxError("Empty phrase in query")
            return Phrase(words)
        if tok.kind == "WORD":
            self.consume()
            words = list(tokenize(tok.value))
            if not words:
                raise QuerySyntaxError(f"Invalid search term: {tok.value!r}")
            node: QueryNode = Term(words[0])
            for w in words[1:]:
                node = And(node, Term(w))
            return node
        raise QuerySyntaxError(f"Unexpected token {tok.value!r}")


def parse_query(raw_query: str) -> QueryNode:
    return _Parser(_lex(raw_query)).parse()


def extract_positive_terms(node: QueryNode) -> list[str]:
    if isinstance(node, Term):
        return [node.value]
    if isinstance(node, Phrase):
        return list(node.terms)
    if isinstance(node, (And, Or)):
        return extract_positive_terms(node.left) + extract_positive_terms(node.right)
    return []
