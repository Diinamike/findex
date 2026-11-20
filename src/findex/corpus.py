from collections.abc import Iterator
from pathlib import Path

Document = tuple[str, str]


def iter_documents(corpus_dir: str | Path) -> Iterator[Document]:
    """Лінивий ітератор: читає по одному текстовому файлу з диска."""
    path = Path(corpus_dir)
    for file_path in sorted(path.rglob("*.txt")):
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                yield file_path.name, file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue


def read_all_documents_eager(corpus_dir: str | Path) -> list[Document]:
    """Жадібна версія: читає весь корпус у RAM одномоментно."""
    return list(iter_documents(corpus_dir))
