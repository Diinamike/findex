content = """# findex — Лабораторна 3: Рейтинг і модель об’єктів (dunder-методи, протоколи, декоратори)

Продовження розробки пошукового рушія `findex`: перетворення інвертованого індексу на повноцінний об’єкт Python (`collections.abc.Mapping`), парсер мови запитів методом рекурсивного спуску, взаємозамінні алгоритми ранжування (**TF-IDF** та **Okapi BM25**) через протоколи, кешування запитів через `@lru_cache`, декоратор `@timed` та генерація сніпетів.

## 1. Модель об’єктів Python: `Index(Mapping)`, дескриптори та `open_index`

Інвертований індекс реалізовано як клас `Index(collections.abc.Mapping)` (`src/findex/index.py`) на корпусі з **1 000 книг Project Gutenberg** (`63 591 631` токенів, `435 397` унікальних термів, середня довжина документа `avg_doc_length = 63 591.6` токенів):

- **Dunder-методи та протокол відображення:** реалізовано `__len__`, `__contains__`, `__getitem__`, `__iter__`, `__repr__`. Успадкування від `collections.abc.Mapping` автоматично надає методи `.keys()`, `.values()`, `.items()`, `.get()`, `__eq__` без ризику мутації внутрішнього стану (на відміну від прямого наслідування `dict`).
- **Дескриптор `@cached_property`:** середня довжина документа `avg_doc_length` обчислюється ліниво при першому зверненні та записується безпосередньо в `instance.__dict__`, забезпечуючи доступ за `O(1)` при наступних викликах у формулі BM25.
- **Менеджер контексту `open_index(path)`:** реалізований через `@contextmanager` із блоком `try ... finally`, що гарантовано викликає `index.close()` і звільняє внутрішні структури навіть при виникненні винятку під час пошуку.
- **Підтримка позицій за прапорцем (`store_positions=True`):** зберігає позиції входжень кожного терма для точного фразового пошуку (`"baker street"`).

## 2. Парсер мови запитів (рекурсивний спуск) та AST

У модулі `src/findex/query.py` реалізовано лексер і парсер рекурсивного спуску для складених булевих та фразових запитів:

- **Граматика з урахуванням пріоритетів:** дужки `(...)` -> унарний `NOT` -> `AND` (включно з неявним `AND` між сусідніми термами) -> `OR`.
- **AST-вузли (`Term`, `Phrase`, `And`, `Or`, `Not`):** незмінні (`frozen=True`) датакласи з підтримкою операторів `&`, `|`, `~` та поліморфного методу `evaluate(index) -> set[int]`.

## 3. Ранжування документів: Протокол `Scorer`, TF-IDF та BM25

У `src/findex/scoring.py` описано структурний протокол `Scorer` (`typing.Protocol`), що дозволяє підміняти моделі оцінювання без зміни логіки `SearchEngine`. Результати повертаються у вигляді `@dataclass(order=True) SearchResult` і відбираються за топ-k через `heapq.nlargest`.

### Перевірки адекватності (Sanity-Check) на корпусі Project Gutenberg (`BM25(k1=1.5, b=0.75)`)

| Перевірка | Умова тесту на корпусі | Отримані значення | Статус |
|-----------|------------------------|-------------------|--------|
| **1. Рідкий термін вище частого** | `tf=1`, середня довжина документа (`63 591` токенів) | `score("sherlock") = 3.6701 > score("time") = 0.0105` | `True` |
| **2. Насичення частоти (20-те повторення)** | Приріст бала для `"sherlock"` на 1-му vs 20-му входженні | 1-ше: `+3.6701`, 20-те: `+0.0312` (< 1% від першого) | `True` |
| **3. Нормалізація довжини документа** | `tf=1` для `"sherlock"` у короткому vs довгому документі | `short(1 000) = 6.5881 > long(200 000) = 1.8675` | `True` |

### Оцінка якості пошуку: `Precision@5` (TF-IDF проти BM25)

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
| 9 | `don quixote sancho` | `Don_Quijote_2000.txt` *(іспаномовний текст у корпусі: Quijote)* | 0.00 | 0.00 |
| 10 | `clausewitz war strategy` | `On_War_1946.txt` | 1.00 | 1.00 |
| **Середнє** | **10 запитів** | — | **0.90** | **0.90** |

## 4. Декоратори `@timed`, `@lru_cache` та сніпети

- **`@timed` (`functools.wraps`):** заміряє час побудови індексу, завантаження з диска та виконання пошуку зі збереженням оригінальних метаданих функцій.
- **`@lru_cache`:** кешує відображення `нормалізований запит -> tuple[doc_id, ...]`, завдяки чому повторний пошук або перемикання `BM25` / `TfIdf` на тому самому запиті не виконує повторний парсинг AST та обхід постінг-листів.
- **Сніпети (`±80` символів):** знаходять вікно з максимальною кількістю унікальних термів запиту та підсвічують збіги у квадратних дужках `[term]`.

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
5. Використання
# 1. Запуск усіх модульних тестів (ЛР 1–3)
uv run pytest -v

# 2. Побудова позиційного індексу, sanity-check рейтингу та демонстрація @timed / @lru_cache
uv run python scripts/evaluate_lab03.py

# 3. Розрахунок порівняльної таблиці Precision@5 (TF-IDF vs BM25)
uv run python scripts/benchmark_p5.py
with open("README.md", "w", encoding="utf-8") as f:
f.write(content)
print("Lines in README.md:", len(content.splitlines()))
