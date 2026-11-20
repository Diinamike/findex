from __future__ import annotations

import logging
import math
import re
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from findex.index import Index
from findex.query import QuerySyntaxError
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchEngine
from findex.web.config import Settings, get_settings
from findex.web.schemas import (
    DocumentOut,
    ErrorResponse,
    HealthResponse,
    ScorerChoice,
    SearchHitOut,
    SearchResponse,
    StatsResponse,
    TermFreqOut,
)

logger = logging.getLogger("findex.web")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _highlight_snippet(snippet: str) -> Markup:
    escaped = escape(snippet)
    highlighted = re.sub(r"\[([^\[\]]+)\]", r"<mark>\1</mark>", str(escaped))
    return Markup(highlighted)


def _select_scorer(name: ScorerChoice) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()


def get_optional_index(request: Request) -> Index | None:
    return cast(Index | None, getattr(request.app.state, "index", None))


def get_index(index: Annotated[Index | None, Depends(get_optional_index)]) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return index


def get_optional_engine(request: Request, index: Annotated[Index | None, Depends(get_optional_index)]) -> SearchEngine | None:
    if index is None:
        return None
    return get_engine(request, index)


def get_engine(request: Request, index: Annotated[Index, Depends(get_index)]) -> SearchEngine:
    cached_engine = cast(SearchEngine | None, getattr(request.app.state, "engine", None))
    if cached_engine is not None and cached_engine.index is index:
        return cached_engine
    engine = SearchEngine(index)
    request.app.state.engine = engine
    return engine


def _execute_search(
    engine: SearchEngine,
    q: str,
    k: int,
    scorer: ScorerChoice,
    page: int,
) -> SearchResponse:
    try:
        all_hits = engine.search(
            q,
            top_k=page * k,
            scorer=_select_scorer(scorer),
            with_snippets=True,
        )
    except QuerySyntaxError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid search query: {exc}") from exc

    total_hits = len(engine.candidate_doc_ids(q))
    total_pages = max(1, math.ceil(total_hits / k)) if total_hits > 0 else 0
    start = (page - 1) * k
    page_slice = all_hits[start : start + k]

    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        page_size=k,
        total_hits=total_hits,
        total_pages=total_pages,
        results=[
            SearchHitOut(
                doc_id=item.doc_id,
                doc_name=item.doc_name,
                score=round(item.score, 6),
                snippet=item.snippet,
            )
            for item in page_slice
        ],
    )


def create_app(
    settings: Settings | None = None,
    *,
    use_async_blocking: bool = False,
) -> FastAPI:
    active_settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.index = None
        app.state.engine = None
        if active_settings.index_path is not None:
            idx_path = Path(active_settings.index_path)
            if idx_path.exists():
                logger.info("Loading search index from %s", idx_path)
                loaded = Index.load(idx_path)
                app.state.index = loaded
                app.state.engine = SearchEngine(loaded)
                logger.info("Loaded index with %d documents", loaded.num_docs)
            else:
                logger.warning("Configured INDEX_PATH does not exist: %s", idx_path)
        try:
            yield
        finally:
            idx = cast(Index | None, getattr(app.state, "index", None))
            if idx is not None:
                idx.close()
                app.state.index = None
                app.state.engine = None

    app = FastAPI(
        title="findex Web Search API",
        version="0.7.0",
        lifespan=lifespan,
    )

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["highlight_snippet"] = _highlight_snippet

    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = cast(str, getattr(request.state, "request_id", "unknown"))
        logger.exception("Unhandled error [request_id=%s] on %s %s", req_id, request.method, request.url.path)
        payload = ErrorResponse(detail="Internal Server Error", request_id=req_id)
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.get("/health", response_model=HealthResponse)
    def health_check(
        response: Response,
        index: Annotated[Index | None, Depends(get_optional_index)],
    ) -> HealthResponse:
        if index is None:
            response.status_code = 503
            return HealthResponse(
                status="unavailable",
                index_loaded=False,
                num_docs=0,
                detail="Index is not loaded",
            )
        return HealthResponse(
            status="ok",
            index_loaded=True,
            num_docs=index.num_docs,
            detail="Ready",
        )

    if use_async_blocking:

        @app.get("/search", response_model=SearchResponse)
        async def search_async_blocking(
            engine: Annotated[SearchEngine, Depends(get_engine)],
            q: Annotated[str, Query(min_length=1, max_length=500)],
            k: Annotated[int, Query(ge=1, le=100)] = 10,
            scorer: ScorerChoice = "bm25",
            page: Annotated[int, Query(ge=1, le=1000)] = 1,
        ) -> SearchResponse:
            return _execute_search(engine, q=q, k=k, scorer=scorer, page=page)

    else:

        @app.get("/search", response_model=SearchResponse)
        def search_threadpool(
            engine: Annotated[SearchEngine, Depends(get_engine)],
            q: Annotated[str, Query(min_length=1, max_length=500)],
            k: Annotated[int, Query(ge=1, le=100)] = 10,
            scorer: ScorerChoice = "bm25",
            page: Annotated[int, Query(ge=1, le=1000)] = 1,
        ) -> SearchResponse:
            return _execute_search(engine, q=q, k=k, scorer=scorer, page=page)

    @app.get("/docs/{doc_id}", response_model=DocumentOut)
    def get_document(
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentOut:
        try:
            meta = index.get_doc(doc_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found") from exc

        text = ""
        if Path(meta.path).exists():
            text = index.read_doc_text(doc_id)

        return DocumentOut(
            doc_id=meta.doc_id,
            doc_name=meta.name,
            length=meta.length,
            path=meta.path,
            text=text,
        )

    @app.get("/stats", response_model=StatsResponse)
    def get_stats(
        index: Annotated[Index, Depends(get_index)],
        top: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> StatsResponse:
        term_dfs = sorted(
            ((term, len(index[term])) for term in index),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top]
        total_tokens = round(index.avg_doc_length * index.num_docs)
        return StatsResponse(
            num_docs=index.num_docs,
            vocab_size=len(index),
            total_tokens=total_tokens,
            avg_doc_length=round(index.avg_doc_length, 2),
            top_terms=[TermFreqOut(term=term, df=df) for term, df in term_dfs],
        )

    @app.get("/", response_class=HTMLResponse)
    def home_page(
        request: Request,
        index: Annotated[Index | None, Depends(get_optional_index)],
        engine: Annotated[SearchEngine | None, Depends(get_optional_engine)],
        q: str = "",
        scorer: ScorerChoice = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        clean_q = q.strip()
        results: list[SearchHitOut] = []
        total_hits = 0
        total_pages = 0
        error_message = ""

        if clean_q:
            if index is None or engine is None:
                error_message = "Індекс не завантажено. Вкажіть змінну оточення INDEX_PATH."
            else:
                try:
                    resp = _execute_search(
                        engine,
                        q=clean_q,
                        k=active_settings.page_size,
                        scorer=scorer,
                        page=page,
                    )
                    results = resp.results
                    total_hits = resp.total_hits
                    total_pages = resp.total_pages
                except HTTPException as exc:
                    error_message = str(exc.detail)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "index_loaded": index is not None,
                "num_docs": index.num_docs if index is not None else 0,
                "q": clean_q,
                "scorer": scorer,
                "page": page,
                "total_hits": total_hits,
                "total_pages": total_pages,
                "results": results,
                "error_message": error_message,
            },
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def document_page(
        request: Request,
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
        q: str = "",
        scorer: ScorerChoice = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        doc = get_document(doc_id=doc_id, index=index)
        return templates.TemplateResponse(
            request=request,
            name="doc.html",
            context={
                "doc": doc,
                "back_q": q.strip(),
                "scorer": scorer,
                "page": page,
            },
        )

    return app


app = create_app()


def create_async_app() -> FastAPI:
    return create_app(use_async_blocking=True)
