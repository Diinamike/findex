import time
from pathlib import Path
from findex.indexer import InvertedIndex, benchmark_memory_variants
from findex.search_engine import BooleanSearchEngine

corpus_dir = Path("data/corpus")

# 1. Бенчмарк пам'яті постінгів
mem_results = benchmark_memory_variants(corpus_dir)
print("\nТаблиця 1: Використання пам'яті варіантами зберігання постінгів")
for name, ram, t in mem_results:
    print(f"{name:32} | {ram:8.2f} MB | {t:6.2f} c")

# 2. Будування індексу для серіалізації
print("\nБудування основного індексу...")
idx = InvertedIndex(with_positions=False)
idx.build_from_directory(corpus_dir)

# Тест pickle
t0 = time.perf_counter()
idx.save_pickle(Path("index.pkl"))
t_save_pkl = time.perf_counter() - t0
size_pkl = Path("index.pkl").stat().st_size / (1024 * 1024)

t0 = time.perf_counter()
InvertedIndex.load_pickle(Path("index.pkl"))
t_load_pkl = time.perf_counter() - t0

# Тест binary
t0 = time.perf_counter()
idx.save_binary(Path("index.bin"))
t_save_bin = time.perf_counter() - t0
size_bin = Path("index.bin").stat().st_size / (1024 * 1024)

t0 = time.perf_counter()
InvertedIndex.load_binary(Path("index.bin"))
t_load_bin = time.perf_counter() - t0

print("\nТаблиця 2: Порівняння серіалізації")
print(f"Pickle | Розмір: {size_pkl:.2f} MB | Save: {t_save_pkl:.2f} c | Load: {t_load_pkl:.2f} c")
print(f"Binary | Розмір: {size_bin:.2f} MB | Save: {t_save_bin:.2f} c | Load: {t_load_bin:.2f} c")

# 3. Бенчмарк Merge vs Set
# Два найчастіших: the, of
# Два рідкісних: (беремо унікальні слова)
frequent_query = "the of"
rare_query = "gutenberg zoroaster"

search_merge = BooleanSearchEngine(idx, engine="merge")
search_set = BooleanSearchEngine(idx, engine="set")

def run_bench(engine, q, iterations=100):
    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.search(q)
    return (time.perf_counter() - t0) / iterations * 1000

m_freq = run_bench(search_merge, frequent_query)
s_freq = run_bench(search_set, frequent_query)
m_rare = run_bench(search_merge, rare_query)
s_rare = run_bench(search_set, rare_query)

print("\nТаблиця 3: Двигун пошуку (час одного запиту в мс)")
print(f"Запит '{frequent_query}' (Часті)   | Merge: {m_freq:.4f} мс | Set: {s_freq:.4f} мс")
print(f"Запит '{rare_query}' (Рідкісні) | Merge: {m_rare:.4f} мс | Set: {s_rare:.4f} мс")
