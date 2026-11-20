from pathlib import Path

import pytest

from findex.index import Index

SAMPLE_DOCS = {
    "doc_sherlock.txt": (
        "Sherlock Holmes lived at Baker Street in London time after time. "
        "Holmes solved mysteries with logic and observation."
    ),
    "doc_python.txt": (
        "Python async and await simplify concurrent programming in time. "
        "The event loop schedules coroutines without Java threads."
    ),
    "doc_short.txt": "Sherlock rare clue.",
    "doc_long.txt": "time " * 250 + "sherlock",
}


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for name, content in SAMPLE_DOCS.items():
        (corpus_dir / name).write_text(content, encoding="utf-8")
    return corpus_dir


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    return Index.build(sample_corpus, store_positions=True)
