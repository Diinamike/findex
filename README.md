# findex — Лабораторна 4: Типи, тести, пакування і CLI

[![CI](https://github.com/Diinamike/findex/actions/workflows/ci.yml/badge.svg?branch=lab-04-branch)](https://github.com/Diinamike/findex/actions/workflows/ci.yml)

Повнотекстова пошукова система `findex` (версія `0.4.0`), оформлена як інстальований Python-пакет із суворою типізацією (`pyright --strict`), 33 тестами (`pytest` + `Hypothesis`), CLI на базі `typer` та `rich`, структурованим логуванням у `stderr` і автоматизованим CI у GitHub Actions.

## 1. Статична типізація (`pyright --strict`) та архітектура

- У `pyproject.toml` увімкнено `typeCheckingMode = "strict"` — перевірка проходить із **0 помилками** та без жодного `# type: ignore`.
- `Scorer` (`src/findex/scoring.py`) реалізовано як `@runtime_checkable class Scorer(Protocol)` — `TfIdf` і `BM25` проходять перевірку сумісності структурно без успадкування.
- Потокові генератори (`iter_documents`, `tokenize`) типізовано як `Iterator[...]`, менеджер контексту `open_index` — як `Generator[Index]`.
- Варіантні рядкові параметри типізовано через `Literal["bm25", "tfidf"]` та `Literal["set", "merge"]`.

## 2. Встановлення та CLI (`typer` + `rich`)

```bash
# Встановлення пакета у глобальне середовище користувача (кладе findex у PATH)
uv tool install .

# 1. Побудова інвертованого індексу зі смугою прогресу rich
findex index data/corpus --out index.bin

# 2. Пошук із форматованою таблицею rich (або чистий JSON у stdout для перенаправлення)
findex search index.bin '(sherlock OR holmes) AND "baker street" NOT watson' --top-k 5
findex search index.bin 'sherlock' --scorer tfidf --engine merge --json > results.json

# 3. Статистика індексу та топ найчастіших термів
findex stats index.bin --top 10

# 4. Діагностичні логи в stderr (-v -> INFO, -vv -> DEBUG)
findex -v search index.bin 'sherlock'
```bash
cat << 'EOF' >> README.md

## 3. Тестування (`pytest` + `Hypothesis`) та покриття коду

Набір із **33 тестів** (`tests/test_tokenizer.py`, `tests/test_lab04.py`, фікстури у `tests/conftest.py`) перевіряє:
- токенізатор (`@pytest.mark.parametrize`, Unicode NFKC-лігатури, апострофи, дефіси, кирилиця);
- збірку індексу, dunder-методи `Mapping`, `@cached_property` та звільнення ресурсів у `open_index` при винятку;
- парсер запитів (порівняння AST-дерев `Term`, `Phrase`, `And`, `Or`, `Not` та синтаксичні помилки);
- протокол `Scorer`, `TfIdf`, `BM25` (усі 3 sanity-check із ЛР 3), серіалізацію `save` / `load` і CLI через `typer.testing.CliRunner`;
- **3 властивості Hypothesis**: `load(save(index))` відновлює ідентичний індекс; постінги завжди відсортовані за зростанням `doc_id`; `merge_postings` збігається з перетином `set`;
- повільні інтеграційні тести виділено маркером `@pytest.mark.slow`.

### Звіт покриття (`uv run pytest --cov=findex --cov-report=term-missing`)

| Модуль | Покриття | Примітка |
|--------|----------|----------|
| `src/findex/__init__.py` | **100%** | Публічний API пакета |
| `src/findex/tokenizer.py` | **100%** | Потоковий Unicode-токенізатор |
| `src/findex/utils.py` | **100%** | Декоратор `@timed` (`ParamSpec`) |
| `src/findex/index.py` | **98%** | `Index(Mapping)`, бінарний формат, `merge_postings` |
| `src/findex/scoring.py` | **92%** | Протокол `Scorer`, `TfIdf`, `BM25` |
| `src/findex/search.py` | **92%** | `SearchEngine`, `@lru_cache`, сніпети |
| `src/findex/query.py` | **89%** | Лексер, рекурсивний спуск, AST-вузли |
| `src/findex/cli.py` | **88%** | Команди `index`, `search`, `stats`, обробка помилок |
| `src/findex/stats.py` | **0%** | Допоміжний бенчмарк Lazy vs Eager з ЛР 1 (навмисно поза CLI-покриттям) |
| **Загальне покриття** | **87%** | **33 passed (603 / 696 statements)** |

**Що навмисно залишено поза покриттям (13%):**
1. Модуль `src/findex/stats.py` (43 рядки) — бенчмарк порівняння лінивого та жадібного конвеєрів із Лабораторної 1, який не входить до рантайму CLI `findex`.
2. Низькорівневі гілки помилок файлової системи (`OSError` / `PermissionError` при читанні битих файлів у `corpus.py` та `cli.py`) та захисні `NotImplementedError` у базовому класі `QueryNode`.
