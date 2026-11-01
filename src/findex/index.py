import argparse
import sys
import time
import tracemalloc
from pathlib import Path

from findex.indexer import InvertedIndex


def main():
    parser = argparse.ArgumentParser(description="Будування та збереження інвертованого індексу findex")
    parser.add_argument("corpus_path", type=Path, help="Шлях до каталогу з корпусом текстів")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Шлях до файлу виводу (.bin або .pkl)")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів у постінгах")
    parser.add_argument("--format", choices=["bin", "pickle"], default=None, help="Формат серіалізації")
    args = parser.parse_args()

    tracemalloc.start()
    t_start = time.perf_counter()

    fmt = args.format or ("pickle" if args.out.suffix == ".pkl" else "bin")
    print(f"Індексація каталогу: {args.corpus_path} (формат: {fmt}, positions={args.positions})...")

    index = InvertedIndex(with_positions=args.positions)
    index.build_from_directory(args.corpus_path)

    t_build = time.perf_counter() - t_start
    peak_ram = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    t_save_start = time.perf_counter()
    if fmt == "pickle":
        index.save_pickle(args.out)
    else:
        index.save_binary(args.out)
    t_save = time.perf_counter() - t_save_start

    file_size_mb = args.out.stat().st_size / (1024 * 1024)

    print("=" * 60)
    print(f"Індексовано документів : {index.total_docs}")
    print(f"Унікальних термів      : {len(index.postings)}")
    print(f"Час побудови індексу   : {t_build:.2f} c")
    print(f"Пікова пам'ять RAM     : {peak_ram:.2f} MB")
    print(f"Час збереження         : {t_save:.2f} c")
    print(f"Розмір файлу індексу   : {file_size_mb:.2f} MB ({args.out})")
    print("=" * 60)
    tracemalloc.stop()


if __name__ == "__main__":
    main()
