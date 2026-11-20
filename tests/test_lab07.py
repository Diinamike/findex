from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx
from fastapi.testclient import TestClient

from findex.query import QuerySyntaxError
from findex.scoring import Scorer, SearchResult
from findex.web.app import create_app, get_engine, get_optional_engine, get_optional_index
from findex.web.config import Settings


@dataclass(frozen=True)
class FakeDocMeta:
    doc_id: int
    name: str
    length: int
    path: str


class FakeIndex:
    def __init__(self) -> None:
        self._docs: dict[int, FakeDocMeta] = {
            0: FakeDocMeta(0, "https://docs.python.org/3/library/asyncio-task.html", 120, "/nonexistent/0"),
            1: FakeDocMeta(1, "https://docs.python.org/3/library/asyncio-eventloop.html", 95, "/nonexistent/1"),
            2: FakeDocMeta(2, "https://docs.python.org/3/library/fastapi-intro.html", 80, "/nonexistent/2"),
        }
        self._postings: dict[str, list[tuple[int, int, list[int]]]] = {
            "taskgroup": [(0, 3, [1, 10, 40])],
            "semaphore": [(0, 2, [5, 22]), (1, 1, [14])],
            "asyncio": [(0, 4, [0, 3, 12, 50]), (1, 5, [2, 8, 19, 30, 44]), (2, 1, [9])],
        }

    @property
    def num_docs(self) -> int:
        return len(self._docs)

    @property
    def avg_doc_length(self) -> float:
        return sum(d.length for d in self._docs.values()) / len(self._docs)

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self):
        return iter(self._postings)

    def __getitem__(self, term: str) -> list[tuple[int, int, list[int]]]:
        return self._postings[term]

    def get_doc(self, doc_id: int) -> FakeDocMeta:
        return self._docs[doc_id]


class FakeSearchEngine:
    def __init__(self, index: FakeIndex, cpu_delay_sec: float = 0.0) -> None:
        self.index = index
        self.cpu_delay_sec = cpu_delay_sec

    def candidate_doc_ids(self, raw_query: str) -> tuple[int, ...]:
        if raw_query.strip() == "asyncio":
            return (0, 1, 2)
        if raw_query.strip() == "TaskGroup AND Semaphore":
            return (0,)
        return ()

    def search(
        self,
        raw_query: str,
        top_k: int = 10,
        scorer: Scorer | None = None,
        with_snippets: bool = True,
    ) -> list[SearchResult]:
        if self.cpu_delay_sec > 0:
            # Імітуємо CPU-bound обчислення, що утримує GIL / потік без await
            deadline = time.perf_counter() + self.cpu_delay_sec
            while time.perf_counter() < deadline:
                pass

        if raw_query.strip() == "AND bad":
            raise QuerySyntaxError("Unexpected operator AND")
        if raw_query.strip() == "BOOM":
            raise RuntimeError("Simulated secret internal failure")

        if raw_query.strip() == "asyncio":
            all_items = [
                SearchResult(2.41, 1, self.index.get_doc(1).name, "The [asyncio] event loop runs coroutines."),
                SearchResult(1.95, 0, self.index.get_doc(0).name, "Use TaskGroup in [asyncio] programs."),
                SearchResult(0.88, 2, self.index.get_doc(2).name, "FastAPI integrates with [asyncio]."),
            ]
            return all_items[:top_k]

        if raw_query.strip() == "TaskGroup AND Semaphore":
            return [
                SearchResult(
                    3.75,
                    0,
                    self.index.get_doc(0).name,
                    "[TaskGroup] coordinates tasks while [Semaphore] bounds concurrency.",
                )
            ][:top_k]

        return []


def test_health_503_when_index_missing() -> None:
    app = create_app(Settings(index_path=None))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["index_loaded"] is False


def test_api_routes_with_dependency_overrides() -> None:
    app = create_app(Settings(index_path=None, page_size=2))
    fake_index = FakeIndex()
    fake_engine = FakeSearchEngine(fake_index)

    app.dependency_overrides[get_optional_index] = lambda: fake_index
    app.dependency_overrides[get_engine] = lambda: fake_engine
    app.dependency_overrides[get_optional_engine] = lambda: fake_engine

    with TestClient(app, raise_server_exceptions=False) as client:
        # 1. /health -> 200
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["num_docs"] == 3

        # 2. Успішний пошук з пагінацією (сторінка 1 і 2 по k=2)
        s1 = client.get("/search", params={"q": "asyncio", "k": 2, "page": 1, "scorer": "bm25"})
        assert s1.status_code == 200
        p1 = s1.json()
        assert p1["total_hits"] == 3
        assert p1["total_pages"] == 2
        assert len(p1["results"]) == 2
        assert p1["results"][0]["doc_id"] == 1

        s2 = client.get("/search", params={"q": "asyncio", "k": 2, "page": 2, "scorer": "tfidf"})
        assert s2.status_code == 200
        assert len(s2.json()["results"]) == 1

        # 3. Валідація -> 422 (порожній q, невалідний k, синтаксична помилка запиту)
        assert client.get("/search", params={"q": ""}).status_code == 422
        assert client.get("/search", params={"q": "asyncio", "k": 0}).status_code == 422
        assert client.get("/search", params={"q": "asyncio", "scorer": "unknown"}).status_code == 422
        assert client.get("/search", params={"q": "AND bad"}).status_code == 422

        # 4. Документи -> 200 і 404
        d0 = client.get("/docs/0")
        assert d0.status_code == 200
        assert d0.json()["doc_id"] == 0
        assert client.get("/docs/999").status_code == 404

        # 5. Статистика -> 200
        st = client.get("/stats", params={"top": 2})
        assert st.status_code == 200
        assert st.json()["num_docs"] == 3
        assert len(st.json()["top_terms"]) == 2

        # 6. Несподіваний виняток -> 500 без витоку стектрейсу, з request_id
        err = client.get("/search", params={"q": "BOOM"}, headers={"X-Request-ID": "req-test-77"})
        assert err.status_code == 500
        err_body = err.json()
        assert err_body == {"detail": "Internal Server Error", "request_id": "req-test-77"}
        assert "RuntimeError" not in err.text
        assert "secret internal failure" not in err.text

        # 7. HTML-інтерфейс Jinja2 (GET / та GET /doc/0)
        html_home = client.get("/", params={"q": "TaskGroup AND Semaphore"})
        assert html_home.status_code == 200
        assert "<mark>TaskGroup</mark>" in html_home.text
        assert "<mark>Semaphore</mark>" in html_home.text

        html_doc = client.get("/doc/0", params={"q": "TaskGroup AND Semaphore"})
        assert html_doc.status_code == 200
        assert "asyncio-task.html" in html_doc.text


def test_def_vs_async_def_concurrent_search_event_loop_blocking() -> None:
    """Порівнюємо 20 одночасних запитів на /search у варіанті def (threadpool) vs async def (blocking)."""

    async def measure(use_async_blocking: bool) -> tuple[float, float]:
        app = create_app(Settings(index_path=None), use_async_blocking=use_async_blocking)
        fake_index = FakeIndex()
        # Кожен пошук займає 15 мс CPU-часу (20 запитів = 300 мс сумарної роботи)
        fake_engine = FakeSearchEngine(fake_index, cpu_delay_sec=0.015)
        app.dependency_overrides[get_optional_index] = lambda: fake_index
        app.dependency_overrides[get_engine] = lambda: fake_engine
        app.dependency_overrides[get_optional_engine] = lambda: fake_engine

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            t0 = time.perf_counter()
            search_tasks = [
                asyncio.create_task(client.get("/search", params={"q": "asyncio"}))
                for _ in range(20)
            ]
            await asyncio.sleep(0)
            th0 = time.perf_counter()
            health_resp = await client.get("/health")
            health_latency_ms = (time.perf_counter() - th0) * 1000
            responses = await asyncio.gather(*search_tasks)
            total_ms = (time.perf_counter() - t0) * 1000

        assert health_resp.status_code == 200
        assert all(r.status_code == 200 for r in responses)
        return total_ms, health_latency_ms

    def_total_ms, def_health_ms = asyncio.run(measure(use_async_blocking=False))
    async_total_ms, async_health_ms = asyncio.run(measure(use_async_blocking=True))

    print(
        f"\n[20 concurrent /search] def (threadpool): total={def_total_ms:.1f}ms, /health={def_health_ms:.1f}ms | "
        f"async def (blocking): total={async_total_ms:.1f}ms, /health={async_health_ms:.1f}ms"
    )
    # У варіанті def /health не чекає завершення всіх 20 CPU-запитів (300 мс), на відміну від async def
    assert def_total_ms < async_total_ms and def_health_ms < async_health_ms
