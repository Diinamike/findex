from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger("findex.crawler")

DEFAULT_USER_AGENT = (
    "findex-crawler/0.6.0 (+https://github.com/Diinamike/findex; contact: tadiana06@gmail.com)"
)

SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".mp3", ".mp4", ".css", ".js",
)


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str:
    """Канонізація URL: нижній регістр схеми/хоста, без #fragment, нормалізований шлях і query."""
    joined = urljoin(base_url, raw_url.strip()) if base_url else raw_url.strip()
    parsed = urlparse(joined)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunparse((scheme, netloc, path, "", sorted_query, ""))


class _LinkAndTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "noscript") and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        self.text_parts.append(cleaned)


def extract_text_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    parser = _LinkAndTextParser()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()

    normalized_links: list[str] = []
    for href in parser.links:
        if href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        canon = canonicalize_url(href, base_url=base_url)
        parsed = urlparse(canon)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.path.lower().endswith(SKIP_EXTENSIONS):
            continue
        normalized_links.append(canon)
    return title, text, normalized_links


class RobotsCache:
    """Кеш правил robots.txt окремо для кожного хоста."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def is_allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        host_key = f"{parsed.scheme}://{parsed.netloc}"
        lock = self._locks.setdefault(host_key, asyncio.Lock())

        async with lock:
            if host_key not in self._parsers:
                rp = RobotFileParser()
                robots_url = f"{host_key}/robots.txt"
                rp.set_url(robots_url)
                try:
                    async with asyncio.timeout(10.0):
                        resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    elif resp.status_code in (401, 403):
                        rp.parse(["User-agent: *", "Disallow: /"])
                    else:
                        rp.parse(["User-agent: *", "Allow: /"])
                except (TimeoutError, httpx.HTTPError, ValueError) as exc:
                    logger.warning("robots.txt fetch failed for %s: %s (allowing)", host_key, exc)
                    rp.parse(["User-agent: *", "Allow: /"])
                self._parsers[host_key] = rp

        return self._parsers[host_key].can_fetch(self.user_agent, url)


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    html: str
    bytes_read: int
    elapsed_sec: float
    attempts: int


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str
    text: str
    status_code: int
    bytes_read: int
    elapsed_sec: float
    depth: int


@dataclass
class CrawlStats:
    start_time: float = field(default_factory=time.perf_counter)
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    active_workers: int = 0
    queue_depth: int = 0
    bytes_total: int = 0

    @property
    def pages_per_sec(self) -> float:
        elapsed = max(1e-6, time.perf_counter() - self.start_time)
        return self.fetched / elapsed


def _parse_retry_after(header_val: str | None) -> float | None:
    if not header_val:
        return None
    try:
        delay = float(header_val.strip())
        return max(0.0, delay)
    except ValueError:
        return None


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_sec: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.4,
    sleep_fn: Callable[[float], object] = asyncio.sleep,
) -> FetchResult:
    """Завантаження сторінки з asyncio.timeout та повторами (429, 5xx, таймаут) + jitter."""
    attempt = 0
    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_sec):
                resp = await client.get(url)
            elapsed = time.perf_counter() - t0
            status = resp.status_code

            if status == 429 or 500 <= status <= 599:
                if attempt <= max_retries:
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    delay = (
                        retry_after
                        if retry_after is not None
                        else backoff_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
                    )
                    logger.warning(
                        "HTTP %s for %s (attempt %d/%d), retrying in %.2fs",
                        status, url, attempt, max_retries + 1, delay,
                    )
                    res = sleep_fn(delay)
                    if asyncio.iscoroutine(res):
                        await res
                    continue
                resp.raise_for_status()

            if 400 <= status <= 499:
                # Інші 4xx (напр. 404, 403) не повторюємо
                resp.raise_for_status()

            raw_bytes = resp.content
            return FetchResult(
                url=url,
                final_url=canonicalize_url(str(resp.url)),
                status_code=status,
                html=resp.text,
                bytes_read=len(raw_bytes),
                elapsed_sec=elapsed,
                attempts=attempt,
            )

        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            if isinstance(exc, httpx.TooManyRedirects):
                logger.error("Redirect loop for %s: %s", url, exc)
                raise
            if attempt <= max_retries:
                delay = backoff_base * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
                logger.warning(
                    "Transient error for %s (%s, attempt %d/%d), retrying in %.2fs",
                    url, type(exc).__name__, attempt, max_retries + 1, delay,
                )
                res = sleep_fn(delay)
                if asyncio.iscoroutine(res):
                    await res
                continue
            raise


async def crawl(
    start_url: str,
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host_limit: int = 4,
    min_delay: float = 0.1,
    timeout_sec: float = 10.0,
    max_retries: int = 3,
    same_domain_only: bool = True,
    allowed_prefix: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> AsyncIterator[CrawledPage]:
    """Асинхронний генератор краулера з чергою, TaskGroup, семафорами та robots.txt."""
    seed = canonicalize_url(start_url)
    seed_parsed = urlparse(seed)
    seed_host = seed_parsed.netloc

    crawl_stats = stats if stats is not None else CrawlStats()
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    out_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    seen: set[str] = {seed}
    queue.put_nowait((seed, 0))
    crawl_stats.queue_depth = queue.qsize()

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_last_req: dict[str, float] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    robots = RobotsCache(user_agent=user_agent)
    stop_event = asyncio.Event()

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    client = httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        max_redirects=5,
        transport=transport,
    )

    async def worker() -> None:
        while not stop_event.is_set():
            try:
                async with asyncio.timeout(0.2):
                    url, depth = await queue.get()
            except TimeoutError:
                if crawl_stats.in_flight == 0 and queue.empty():
                    break
                continue

            crawl_stats.active_workers += 1
            crawl_stats.queue_depth = queue.qsize()
            if stop_event.is_set() or crawl_stats.fetched >= max_pages:
                crawl_stats.active_workers -= 1
                queue.task_done()
                break

            host = urlparse(url).netloc
            h_sem = host_sems.setdefault(host, asyncio.Semaphore(max(1, per_host_limit)))
            h_lock = host_rate_locks.setdefault(host, asyncio.Lock())

            try:
                allowed = await robots.is_allowed(client, url)
                if not allowed:
                    logger.info("DISALLOWED by robots.txt: %s", url)
                    continue

                async with global_sem, h_sem:
                    if stop_event.is_set() or crawl_stats.fetched >= max_pages:
                        continue

                    async with h_lock:
                        now = time.perf_counter()
                        last = host_last_req.get(host, 0.0)
                        wait = min_delay - (now - last)
                        if wait > 0:
                            await asyncio.sleep(wait)
                        host_last_req[host] = time.perf_counter()

                    crawl_stats.in_flight += 1
                    try:
                        res = await fetch_with_retry(
                            client,
                            url,
                            timeout_sec=timeout_sec,
                            max_retries=max_retries,
                        )
                    finally:
                        crawl_stats.in_flight -= 1

                    if stop_event.is_set() or crawl_stats.fetched >= max_pages:
                        continue

                    title, text, links = extract_text_and_links(res.html, res.final_url)
                    crawl_stats.fetched += 1
                    crawl_stats.bytes_total += res.bytes_read
                    logger.info(
                        "OK %d %s (%d bytes, %.3fs, attempts=%d)",
                        res.status_code, res.final_url, res.bytes_read, res.elapsed_sec, res.attempts,
                    )

                    page = CrawledPage(
                        url=res.final_url,
                        title=title,
                        text=text,
                        status_code=res.status_code,
                        bytes_read=res.bytes_read,
                        elapsed_sec=res.elapsed_sec,
                        depth=depth,
                    )
                    await out_queue.put(page)

                    if crawl_stats.fetched >= max_pages:
                        stop_event.set()
                        break

                    for link in links:
                        if link in seen:
                            continue
                        lp = urlparse(link)
                        if same_domain_only and lp.netloc != seed_host:
                            continue
                        if allowed_prefix and not link.startswith(allowed_prefix):
                            continue
                        seen.add(link)
                        queue.put_nowait((link, depth + 1))
                    crawl_stats.queue_depth = queue.qsize()

            except (TimeoutError, httpx.HTTPError, ValueError) as exc:
                crawl_stats.errors += 1
                logger.error("FAIL %s: %s: %s", url, type(exc).__name__, exc)
            finally:
                crawl_stats.active_workers -= 1
                queue.task_done()

    async def runner() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(worker()) for _ in range(max(1, concurrency))]
                await stop_event.wait() if max_pages == 0 else None
                # Чекаємо завершення черги або досягнення max_pages
                while not stop_event.is_set():
                    if queue.empty() and crawl_stats.active_workers == 0:
                        stop_event.set()
                        break
                    await asyncio.sleep(0.05)
                for t in tasks:
                    if not t.done():
                        t.cancel()
        except* asyncio.CancelledError:
            pass
        finally:
            await out_queue.put(None)

    runner_task = asyncio.create_task(runner())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass
        await client.aclose()
