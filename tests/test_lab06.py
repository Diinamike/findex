from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from findex.crawler import (
    RobotsCache,
    canonicalize_url,
    crawl,
    extract_text_and_links,
    fetch_with_retry,
)
from findex.index import Index
from findex.search import SearchEngine


def test_canonicalize_url() -> None:
    u1 = canonicalize_url("HTTPS://Docs.Python.Org:443/3/library/?z=2&a=1#asyncio-tasks")
    u2 = canonicalize_url("https://docs.python.org/3/library?a=1&z=2")
    assert u1 == u2 == "https://docs.python.org/3/library?a=1&z=2"


def test_extract_text_and_links() -> None:
    html = """
    <html>
      <head><title>AsyncIO Guide</title><script>var secret = 1;</script></head>
      <body>
        <h1>Event Loop</h1>
        <p>Coroutines run cooperatively.</p>
        <a href="/3/library/asyncio-task.html#top">Tasks</a>
        <a href="/logo.png">Logo</a>
        <a href="mailto:test@example.com">Mail</a>
      </body>
    </html>
    """
    title, text, links = extract_text_and_links(html, "https://docs.python.org/3/library/asyncio.html")
    assert title == "AsyncIO Guide"
    assert "Event Loop" in text
    assert "secret" not in text
    assert links == ["https://docs.python.org/3/library/asyncio-task.html"]


def test_fetch_retry_429_5xx_and_no_retry_404() -> None:
    async def _run() -> None:
        calls: dict[str, int] = {}
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls[path] = calls.get(path, 0) + 1
            if path == "/rate":
                if calls[path] < 3:
                    return httpx.Response(429, headers={"Retry-After": "0.25"})
                return httpx.Response(200, text="recovered")
            if path == "/missing":
                return httpx.Response(404, text="not found")
            if path == "/loop-a":
                return httpx.Response(302, headers={"Location": "https://example.test/loop-b"})
            if path == "/loop-b":
                return httpx.Response(302, headers={"Location": "https://example.test/loop-a"})
            return httpx.Response(503, text="unavailable")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, follow_redirects=True, max_redirects=3) as client:
            res = await fetch_with_retry(
                client, "https://example.test/rate", max_retries=3, sleep_fn=fake_sleep
            )
            assert res.status_code == 200
            assert res.attempts == 3
            assert slept == [0.25, 0.25]

            with pytest.raises(httpx.HTTPStatusError):
                await fetch_with_retry(
                    client, "https://example.test/missing", max_retries=3, sleep_fn=fake_sleep
                )
            # 404 не повторюється — рівно 1 спроба
            assert calls["/missing"] == 1

            with pytest.raises(httpx.TooManyRedirects):
                await fetch_with_retry(
                    client, "https://example.test/loop-a", max_retries=3, sleep_fn=fake_sleep
                )

    asyncio.run(_run())


def test_fetch_timeout_retries() -> None:
    async def _run() -> None:
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(0.05)
            return httpx.Response(200, text="late")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(TimeoutError):
                await fetch_with_retry(
                    client,
                    "https://example.test/slow",
                    timeout_sec=0.01,
                    max_retries=2,
                    backoff_base=0.001,
                )
        assert attempts == 3

    asyncio.run(_run())


def test_robots_cache_and_end_to_end_crawl_to_index(tmp_path: Path) -> None:
    async def _run() -> Path:
        robots_requests = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal robots_requests
            path = request.url.path
            if path == "/robots.txt":
                robots_requests += 1
                return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
            if path == "/":
                return httpx.Response(
                    200,
                    text=(
                        "<title>Home</title><p>Welcome to async crawler.</p>"
                        "<a href='/p1#s1'>P1</a><a href='/p1#s2'>P1 dup</a>"
                        "<a href='/p2?b=2&a=1'>P2</a><a href='/private/secret'>Secret</a>"
                        "<a href='/broken'>Broken</a>"
                    ),
                )
            if path == "/p1":
                return httpx.Response(200, text="<title>Page One</title><p>TaskGroup and Semaphore.</p>")
            if path == "/p2":
                return httpx.Response(200, text="<title>Page Two</title><p>Exponential backoff jitter.</p>")
            return httpx.Response(404, text="Not found")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rc = RobotsCache()
            assert await rc.is_allowed(client, "https://site.test/p1") is True
            assert await rc.is_allowed(client, "https://site.test/private/secret") is False
            assert robots_requests == 1

        jsonl_path = tmp_path / "crawled.jsonl"
        pages = []
        with jsonl_path.open("w", encoding="utf-8") as f:
            async for page in crawl(
                "https://site.test/",
                max_pages=10,
                concurrency=4,
                min_delay=0.0,
                transport=transport,
            ):
                pages.append(page)
                f.write(json.dumps({"url": page.url, "title": page.title, "text": page.text}) + "\n")

        assert len(pages) == 3
        assert {p.url for p in pages} == {
            "https://site.test/",
            "https://site.test/p1",
            "https://site.test/p2?a=1&b=2",
        }
        return jsonl_path

    jsonl_file = asyncio.run(_run())
    idx = Index.build(jsonl_file, store_positions=True)
    assert idx.num_docs == 3
    engine = SearchEngine(idx)
    hits = engine.search("TaskGroup AND Semaphore", top_k=3)
    assert len(hits) == 1
    assert hits[0].doc_name == "https://site.test/p1"
