from __future__ import annotations

import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from findex.index import build_parallel_index, collect_corpus_paths

CORPUS = Path("data/corpus")
WORKERS_LIST = [1, 2, 4, 8, os.cpu_count() or 10]


def get_rss_mb() -> float:
    # На macOS ru_maxrss повертається в байтах
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def measure_once(executor_kind: str, workers: int) -> dict[str, float]:
    gc.collect()
    t_wall0 = time.perf_counter()
    t_cpu0 = time.process_time()
    idx, stats = build_parallel_index(
        CORPUS,
        workers=workers,
        executor_kind=executor_kind,  # type: ignore[arg-type]
        store_positions=True,
    )
    wall = time.perf_counter() - t_wall0
    cpu = time.process_time() - t_cpu0
    rss = get_rss_mb()
    idx.close()
    return {
        "wall": wall,
        "cpu": cpu,
        "rss_mb": rss,
        "merge_ms": stats.merge_time_sec * 1000.0,
    }


def benchmark_executor(executor_kind: str, workers_list: list[int], repeats: int = 3) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for w in workers_list:
        runs = [measure_once(executor_kind, w) for _ in range(repeats)]
        rows.append(
            {
                "workers": float(w),
                "wall": statistics.median(r["wall"] for r in runs),
                "cpu": statistics.median(r["cpu"] for r in runs),
                "rss_mb": max(r["rss_mb"] for r in runs),
                "merge_ms": statistics.median(r["merge_ms"] for r in runs),
            }
        )
    return rows


def demo_race_condition() -> dict[str, int]:
    """Демонстрація втрати оновлень без Lock у free-threaded Python 3.13t."""
    n_threads = 8
    iters = 100_000
    expected = n_threads * iters

    unsafe_counter: dict[str, int] = {"tokens": 0}

    def unsafe_worker() -> None:
        for _ in range(iters):
            cur = unsafe_counter["tokens"]
            unsafe_counter["tokens"] = cur + 1

    threads = [threading.Thread(target=unsafe_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    safe_counter: dict[str, int] = {"tokens": 0}
    lock = threading.Lock()

    def safe_worker() -> None:
        for _ in range(iters):
            with lock:
                safe_counter["tokens"] += 1

    threads_safe = [threading.Thread(target=safe_worker) for _ in range(n_threads)]
    for t in threads_safe:
        t.start()
    for t in threads_safe:
        t.join()

    return {
        "expected": expected,
        "unsafe_actual": unsafe_counter["tokens"],
        "lost_updates": expected - unsafe_counter["tokens"],
        "safe_actual": safe_counter["tokens"],
    }


def benchmark_io_only(workers_list: list[int], repeats: int = 3) -> list[dict[str, float]]:
    paths = collect_corpus_paths(CORPUS)
    out: list[dict[str, float]] = []
    for w in workers_list:
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            if w == 1:
                total_bytes = sum(len(p.read_bytes()) for p in paths)
            else:
                with ThreadPoolExecutor(max_workers=w) as pool:
                    total_bytes = sum(pool.map(lambda p: len(p.read_bytes()), paths))
            assert total_bytes > 0
            times.append(time.perf_counter() - t0)
        out.append({"workers": float(w), "wall": statistics.median(times)})
    return out


def plot_speedup(data: dict[str, object], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    serial_wall = float(data["serial_wall"])  # type: ignore[arg-type]
    workers = [int(r["workers"]) for r in data["processes"]]  # type: ignore[index,attr-defined]
    s_threads = [serial_wall / r["wall"] for r in data["threads_312"]]  # type: ignore[index,attr-defined]
    s_313t = [serial_wall / r["wall"] for r in data["threads_313t"]]  # type: ignore[index,attr-defined]
    s_proc = [serial_wall / r["wall"] for r in data["processes"]]  # type: ignore[index,attr-defined]

    plt.figure(figsize=(8.5, 5.2), dpi=150)
    plt.plot(workers, workers, "--", color="#888888", linewidth=1.5, label="Ideal linear speedup (S = N)")
    plt.plot(workers, s_proc, "o-", color="#1f77b4", linewidth=2.2, label="Processes (Python 3.12, spawn)")
    plt.plot(workers, s_313t, "s-", color="#2ca02c", linewidth=2.2, label="Threads (Python 3.13t, GIL=False)")
    plt.plot(workers, s_threads, "^-", color="#d62728", linewidth=2.0, label="Threads (Python 3.12, GIL=True)")

    plt.title("findex Parallel Indexing Speedup on Apple M5 (10 cores, 1000 docs)")
    plt.xlabel("Number of workers (N)")
    plt.ylabel("Speedup vs serial (T_serial / T_N)")
    plt.xticks(workers)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if "--worker-313t" in sys.argv:
        res = {
            "python_version": sys.version.split()[0],
            "gil_enabled": gil_enabled,
            "threads_313t": benchmark_executor("threads", WORKERS_LIST, repeats=3),
            "race_demo": demo_race_condition(),
        }
        print(json.dumps(res))
        return

    print("1/4 Benchmarking serial (Python 3.12)...")
    serial_rows = benchmark_executor("serial", [1], repeats=3)
    serial_wall = serial_rows[0]["wall"]

    print("2/4 Benchmarking threads (Python 3.12, GIL=True)...")
    threads_312 = benchmark_executor("threads", WORKERS_LIST, repeats=3)

    print("3/4 Benchmarking processes (Python 3.12, spawn)...")
    processes_312 = benchmark_executor("processes", WORKERS_LIST, repeats=3)

    print("4/4 Benchmarking threads (Python 3.13t, GIL=False)...")
    cmd = ["uv", "run", "--python", "3.13t", "--no-project", "python", "scripts/bench_lab05.py", "--worker-313t"]
    env = {**os.environ, "PYTHONPATH": "src"}
    raw_313t = subprocess.check_output(cmd, env=env, text=True)
    data_313t = json.loads(raw_313t.strip().splitlines()[-1])

    io_rows = benchmark_io_only(WORKERS_LIST, repeats=3)

    full_data = {
        "cpu_cores": os.cpu_count(),
        "serial_wall": serial_wall,
        "serial": serial_rows[0],
        "threads_312": threads_312,
        "processes": processes_312,
        "threads_313t": data_313t["threads_313t"],
        "race_demo": data_313t["race_demo"],
        "gil_313t": data_313t["gil_enabled"],
        "io_only": io_rows,
    }
    Path("docs/bench_lab05.json").write_text(json.dumps(full_data, indent=2), encoding="utf-8")
    plot_speedup(full_data, Path("docs/speedup_lab05.png"))
    print("DONE! Saved docs/bench_lab05.json and docs/speedup_lab05.png")


if __name__ == "__main__":
    main()
