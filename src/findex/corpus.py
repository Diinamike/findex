from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

Document = tuple[str, str]


def iter_documents(corpus_source: str | Path) -> Iterator[Document]:
    """Лінивий ітератор: читає по одному .txt файлу з директорії або рядок із .jsonl."""
    path = Path(corpus_source)
    if path.is_file() and path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                raw_obj: object = json.loads(line)
                if not isinstance(raw_obj, dict):
                    continue
                record = cast(dict[str, object], raw_obj)
                url = str(record.get("url") or f"doc_{line_no}")
                title = str(record.get("title") or "").strip()
                text = str(record.get("text") or "")
                full_text = f"{title}\n{text}" if title else text
                yield url, full_text
        return

    for file_path in sorted(path.rglob("*.txt")):
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                yield file_path.name, file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue


def read_all_documents_eager(corpus_dir: str | Path) -> list[Document]:
    """Жадібна версія: читає весь корпус у RAM одномоментно."""
    return list(iter_documents(corpus_dir))
