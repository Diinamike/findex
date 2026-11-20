# findex — Лабораторна 3: Рейтинг і модель об’єктів (dunder-методи, протоколи, декоратори)

Продовження розробки пошукового рушія `findex`: інвертований індекс як об’єкт Python (`collections.abc.Mapping`), парсер мови запитів методом рекурсивного спуску, взаємозамінні алгоритми ранжування (**TF-IDF** та **Okapi BM25**) через протоколи, кешування через `@lru_cache`, `@timed` та сніпети.

## 1. Модель об’єктів Python: `Index(Mapping)`, дескриптори та `open_index`
- `Index(collections.abc.Mapping)` (`src/findex/index.py`) на корпусі з **1 000 книг** (`63 591 631` токенів, `435 397` унікальних термів, `avg_doc_length = 63 591.6`): реалізовано `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__`, `num_docs`, `df(term)` та `@cached_property` для `avg_doc_length`.
- `open_index(path)` (`@contextmanager`): гарантує виклик `index.close()` навіть при винятку.
- Підтримка позицій (`store_positions=True`) для точного фразового пошуку.

## 2. Парсер мови запитів (рекурсивний спуск) та AST
- Граматика (`src/findex/query.py`): `(...)` -> `NOT` -> `AND` (та неявний `AND`) -> `OR`.
- AST-вузли `Term`, `Phrase`, `And`, `Or`, `Not` підтримують `&`, `|`, `~` і `evaluate(index)`.

## 3. Ранжування документів: Протокол `Scorer`, TF-IDF та BM25

### Перевірки адекватності (Sanity-Check) для `BM25(k1=1.5, b=0.75)`
1. **Рідкий термін вище частого** (`tf=1`, середня довжина документа): `score("sherlock") = 3.6701 > score("time") = 0.0105` (`True`).
2. **Насичення частоти (20-те повторення)**: 1-ше входження `+3.6701`, 20-те `+0.0312` (< 1% від першого) (`True`).
3. **Нормалізація довжини документа** (`tf=1` для `"sherlock"`): `short(1 000) = 6.5881 > long(200 000) = 1.8675` (`True`).

### Порівняння якості пошуку: `Precision@5` (TF-IDF vs BM25)

| № | Запит | Очікуваний документ у топ-5 | TF-IDF P@5 | BM25 P@5 |
|---|-------|-----------------------------|------------|----------|
| 1 | `sherlock holmes` | `The_Adventures_of_Sherlock_Holmes_1661.txt` | 1.00 | 1.00 |
| 2 | `hamlet ophelia` | `Hamlet_1787.txt` | 1.00 | 1.00 |
| 3 | `macbeth duncan` | `Macbeth_1795.txt` | 1.00 | 1.00 |
| 4 | `othello desdemona` | `Othello_1793.txt` | 1.00 | 1.00 |
| 5 | `romeo juliet` | `Romeo_and_Juliet_1777.txt` | 1.00 | 1.00 |
| 6 | `darwin natural selection` | `Darwin_and_Modern_Science_1909.txt`, `The_Autobiography_of_Charles_Darwin_2010.txt` | 1.00 | 1.00 |
| 7 | `odysseus penelope` | `The_Odyssey_1727.txt` | 1.00 | 1.00 |
| 8 | `julius caesar brutus` | `Julius_Caesar_1785.txt` | 1.00 | 1.00 |
| 9 | `don quixote sancho` | `Don_Quijote_2000.txt` *(іспанський текст: Quijote)* | 0.00 | 0.00 |
| 10 | `clausewitz war strategy` | `On_War_1946.txt` | 1.00 | 1.00 |
| **Середнє** | **10 запитів** | — | **0.90** | **0.90** |

## 4. Декоратори `@timed`, `@lru_cache` та сніпети (`±80` символів)

```text
[INFO] Index.build executed in 30.7400 s
Index(docs=1000, terms=435397, positions=True, status='open'), avg_doc_length=63591.6
[INFO] Index.load executed in 6.1923 s
Query: (sherlock OR holmes) AND "baker street" NOT watson
[INFO] Query '(sherlock OR holmes) AND "baker street" NOT watson' -> CACHE MISS (hits=0, misses=1)
[INFO] SearchEngine.search executed in 0.0149 s
  [15.7988] Gutenberg_Book_1264_1264.txt: ...machinery, a threshing-machine, say, or one of those hay-making furies. But [Sherlock] [Holmes] (now happily dead) would have fancied nothing of the kind. He...
[INFO] Query '(sherlock OR holmes) AND "baker street" NOT watson' -> CACHE HIT (hits=1, misses=1)
[INFO] SearchEngine.search executed in 0.0110 s
CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)
## 5. Використання
uv run pytest -v
uv run python scripts/evaluate_lab03.py
uv run python scripts/benchmark_p5.py
