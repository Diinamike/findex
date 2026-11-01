import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from findex.corpus import iter_documents, read_all_documents_eager
from findex.tokenizer import tokenize


def compute_stats_lazy(corpus_dir: Path):
    """
    Збір усієї статистики корпусу за ОДИН прохід:
    - Кількість документів
    - Загальна кількість токенів
    - Розмір словника (унікальні терми)
    - Топ-50 найчастіших термів
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count = 0
    token_count = 0
    vocab = Counter()

    # ОДИН ПРОХІД по лінивому конвеєру
    for doc_id, text in iter_documents(corpus_dir):
        doc_count += 1
        for token in tokenize(text):
            token_count += 1
            vocab[token] += 1

    top50 = vocab.most_common(50)
    vocab_size = len(vocab)

    elapsed = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": "Lazy (Generators)",
        "docs": doc_count,
        "tokens": token_count,
        "vocab_size": vocab_size,
        "peak_ram_mb": peak_memory / (1024 * 1024),
        "time_sec": elapsed,
        "top50": top50,
    }


def compute_stats_eager(corpus_dir: Path):
    """
    Навмисно жадібна версія через збереження всіх проміжних списків у пам'яті.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. Завантажуємо всі документи в список
    all_docs = read_all_documents_eager(corpus_dir)
    doc_count = len(all_docs)

    # 2. Збираємо всі токени у велетенський список
    all_tokens = []
    for doc_id, text in all_docs:
        all_tokens.extend(list(tokenize(text)))

    token_count = len(all_tokens)

    # 3. Підраховуємо частоти
    vocab = Counter(all_tokens)
    top50 = vocab.most_common(50)
    vocab_size = len(vocab)

    elapsed = time.perf_counter() - t0
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": "Eager (Lists)",
        "docs": doc_count,
        "tokens": token_count,
        "vocab_size": vocab_size,
        "peak_ram_mb": peak_memory / (1024 * 1024),
        "time_sec": elapsed,
        "top50": top50,
    }


def main():
    corpus_path = Path("data/corpus")
    if not corpus_path.exists():
        print("Помилка: директорія data/corpus не знайдена.")
        sys.exit(1)

    print("Обчислення статистики: Lazy pipeline...")
    lazy = compute_stats_lazy(corpus_path)

    print("Обчислення статистики: Eager pipeline...")
    eager = compute_stats_eager(corpus_path)

    print("\n" + "=" * 65)
    print(f"{'Метрика':<25} | {'Lazy (Generators)':<17} | {'Eager (Lists)':<15}")
    print("-" * 65)
    print(f"{'Кількість документів':<25} | {lazy['docs']:<17} | {eager['docs']:<15}")
    print(f"{'Всього токенів':<25} | {lazy['tokens']:<17} | {eager['tokens']:<15}")
    print(f"{'Розмір словника':<25} | {lazy['vocab_size']:<17} | {eager['vocab_size']:<15}")
    print(f"{'Пікова RAM (MB)':<25} | {lazy['peak_ram_mb']:<17.2f} | {eager['peak_ram_mb']:<15.2f}")
    print(f"{'Час виконання (с)':<25} | {lazy['time_sec']:<17.2f} | {eager['time_sec']:<15.2f}")
    print("=" * 65)

    print("\nТоп-50 найчастіших термів (Lazy):")
    for i, (term, count) in enumerate(lazy["top50"], 1):
        print(f"{i:>2}. {term:<15} : {count}")


if __name__ == "__main__":
    main()
