from pathlib import Path
from typing import Iterator, List, Tuple


def iter_documents(corpus_dir: Path | str) -> Iterator[Tuple[str, str]]:
    """Лінивий генератор, що читає документи по одному з каталогу (включно з підкаталогами)."""
    path = Path(corpus_dir)
    for file_path in sorted(path.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        yield file_path.name, text


def read_all_documents_eager(corpus_dir: Path | str) -> List[Tuple[str, str]]:
    """Жадібне завантаження всього корпусу в список у пам'яті."""
    path = Path(corpus_dir)
    docs = []
    for file_path in sorted(path.rglob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        docs.append((file_path.name, text))
    return docs
