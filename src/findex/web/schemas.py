from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScorerChoice = Literal["bm25", "tfidf"]


class SearchHitOut(BaseModel):
    doc_id: int
    doc_name: str
    score: float
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    scorer: ScorerChoice
    page: int
    page_size: int
    total_hits: int
    total_pages: int
    results: list[SearchHitOut]


class DocumentOut(BaseModel):
    doc_id: int
    doc_name: str
    length: int
    path: str
    text: str


class TermFreqOut(BaseModel):
    term: str
    df: int


class StatsResponse(BaseModel):
    num_docs: int
    vocab_size: int
    total_tokens: int
    avg_doc_length: float
    top_terms: list[TermFreqOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
    num_docs: int = 0
    detail: str = ""


class ErrorResponse(BaseModel):
    detail: str
    request_id: str = Field(default="")
