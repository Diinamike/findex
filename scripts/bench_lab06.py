from __future__ import annotations

import asyncio
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from findex.crawler import CrawlStats, crawl

SEED = "https://docs.python.org/3/library/index.html"
PREFIX = "https://docs.python.org/3/library/"
MAX_PAGES = 200


async def run_single(concurrency: int, out_jsonl: Path) -> dict[str, float | int]:
    stats = CrawlStats()
    t0 = time.perf_counter()
    with out_jsonl.open("w", encoding="utf-8") as jf:
        async for page in crawl(
            SEED,
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host_limit=concurrency,
            min_delay=0.02,
            allowed_prefix=PREFIX,
            stats=stats,
        ):
            jf.write(
                json.dumps({"url": page.url, "title": page.title, "text": page.text}, ensure_ascii=False)
                + "\n"
            )
            jf.flush()
    wall = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    return {
        "concurrency": concurrency,
        "pages": stats.fetched,
        "wall": round(wall, 2),
        "pages_per_sec": round(stats.fetched / max(wall, 1e-6), 2),
        "errors": stats.errors,
        "rss_mb": round(rss_mb, 1),
    }


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        c = int(sys.argv[2])
        res = asyncio.run(run_single(c, Path(f"data/crawled_c{c}.jsonl")))
        print(json.dumps(res))
        return

    results = []
    for c in (1, 5, 20):
        raw = subprocess.check_output(
            [sys.executable, "scripts/bench_lab06.py", "--one", str(c)], text=True
        )
        row = json.loads(raw.strip().splitlines()[-1])
        print(row)
        results.append(row)
    Path("docs/bench_lab06.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
