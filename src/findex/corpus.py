from pathlib import Path
from typing import Generator, List, Tuple, Union

Document = Tuple[str, str]

def iter_documents(corpus_dir: Union[str, Path]) -> Generator[Document, None, None]:
    """Лінивий генератор: читає по одному файлу з диска."""
    path = Path(corpus_dir)
    for file_path in sorted(path.rglob("*.txt")):
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                yield file_path.name, file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

def read_all_documents_eager(corpus_dir: Union[str, Path]) -> List[Document]:
    """Жадібна версія: читає весь корпус у RAM одномоментно."""
    path = Path(corpus_dir)
    docs = []
    for file_path in sorted(path.rglob("*.txt")):
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                docs.append((file_path.name, file_path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, PermissionError):
                continue
    return docs
