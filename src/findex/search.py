import argparse
import sys
import time
import tracemalloc
from pathlib import Path

from findex.indexer import InvertedIndex
from findex.search_engine import BooleanSearchEngine


def main():
    parser = argparse.ArgumentParser(description="Пошук за інвертованим індексом findex")
    parser.add_argument("index_path", type=Path, help="Шлях до індексу (.bin або .pkl)")
    parser.add_argument("query", type=str, help="Булевий запит (наприклад: 'gutenberg AND project', 'lord OR king')")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Алгоритм пошуку (merge або set)")
    parser.add_argument("--limit", type=int, default=10, help="Кількість результатів для виводу")
    args = parser.parse_args()

    tracemalloc.start()
    t_load_start = time.perf_counter()

    if args.index_path.suffix == ".pkl":
        index = InvertedIndex.load_pickle(args.index_path)
    else:
        index = InvertedIndex.load_binary(args.index_path)

    t_load = time.perf_counter() - t_load_start

    searcher = BooleanSearchEngine(index, engine=args.engine)

    t_search_start = time.perf_counter()
    matching_doc_ids = searcher.search(args.query)
    t_search = time.perf_counter() - t_search_start

    peak_ram = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    print("=" * 60)
    print(f"Запит       : {args.query}")
    print(f"Алгоритм    : {args.engine}")
    print(f"Знайдено    : {len(matching_doc_ids)} документів")
    print(f"Час пошуку  : {t_search * 1000:.3f} мс (завантаження індексу: {t_load:.2f} с)")
    print(f"Пікова RAM  : {peak_ram:.2f} MB")
    print("-" * 60)

    for doc_id in matching_doc_ids[: args.limit]:
        meta = index.doc_meta[doc_id]
        print(f"[{doc_id}] {meta.name} (довжина: {meta.length} токенів)")

    if len(matching_doc_ids) > args.limit:
        print(f"... і ще {len(matching_doc_ids) - args.limit} результатів")
    print("=" * 60)


if __name__ == "__main__":
    main()
