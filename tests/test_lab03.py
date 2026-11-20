import pytest

from findex.index import Index, open_index
from findex.query import And, Not, Or, Phrase, Term, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.search import SearchEngine, extract_snippet


@pytest.fixture
def sample_corpus(tmp_path):
    (tmp_path / "doc1.txt").write_text("python async await event loop concurrency", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("python java virtual machine garbage collector", encoding="utf-8")
    (tmp_path / "doc3.txt").write_text(
        "async programming in python uses an event loop and coroutines " + ("filler " * 80),
        encoding="utf-8",
    )
    return tmp_path


def test_index_mapping_and_properties(sample_corpus):
    idx = Index.build(sample_corpus, store_positions=True)

    assert len(idx) > 0
    assert "python" in idx
    assert "nonexistent_term_xyz" not in idx
    assert idx.num_docs == 3
    assert idx.df("python") == 3
    assert idx.df("java") == 1
    assert "python" in list(idx)
    assert "docs=3" in repr(idx)

    # Перевірка cached_property avg_doc_length
    avg1 = idx.avg_doc_length
    assert "avg_doc_length" in idx.__dict__
    assert idx.avg_doc_length == avg1


def test_open_index_context_manager_on_exception(sample_corpus, tmp_path):
    idx_file = tmp_path / "test.bin"
    Index.build(sample_corpus).save(idx_file)

    with pytest.raises(RuntimeError):
        with open_index(idx_file) as idx:
            assert "python" in idx
            raise RuntimeError("boom")

    # Після винятку індекс має бути закритим
    with pytest.raises(ValueError, match="closed Index"):
        _ = idx["python"]

def test_parser_ast_tree_comparison():
    ast = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(
                Term("python"),
                Or(Term("async"), Term("await")),
            ),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert ast == expected

    # Перевірка операторів &, |, ~
    manual = Term("python") & (Term("async") | Term("await")) & ~Term("java") & Phrase(["event", "loop"])
    assert manual == expected


def test_phrase_search_and_without_positions(sample_corpus):
    idx_pos = Index.build(sample_corpus, store_positions=True)
    q = parse_query('"event loop"')
    assert q.evaluate(idx_pos) == {0, 2}

    idx_no_pos = Index.build(sample_corpus, store_positions=False)
    with pytest.raises(ValueError, match="store_positions=True"):
        q.evaluate(idx_no_pos)


def test_scorers_and_lru_cache(sample_corpus):
    idx = Index.build(sample_corpus, store_positions=True)
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()
    assert isinstance(bm25, Scorer)
    assert isinstance(tfidf, Scorer)

    engine = SearchEngine(idx, default_scorer=bm25)
    res1 = engine.search("python async", top_k=2)
    assert engine.cache_info.misses == 1
    assert engine.cache_info.hits == 0

    # Повторний запит має взятися з кешу
    res2 = engine.search("python async", top_k=2, scorer=tfidf)
    assert engine.cache_info.hits == 1
    assert len(res1) == 2 and len(res2) == 2

    # Перевірка SearchResult(order=True)
    assert SearchResult(1.0, 1, "b") < SearchResult(2.0, 0, "a")


def test_snippet_extraction():
    text = "Start of the text. Here python runs an event loop very efficiently. End of text."
    snip = extract_snippet(text, ["python", "loop"], window=40)
    assert "[python]" in snip
    assert "[loop]" in snip
