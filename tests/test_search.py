import pytest
from findex.indexer import InvertedIndex
from findex.models import DocMeta, Posting
from findex.search_engine import BooleanSearchEngine


@pytest.fixture
def mock_index():
    idx = InvertedIndex()
    idx.total_docs = 4
    idx.doc_meta = {
        0: DocMeta(0, "doc0.txt", 10),
        1: DocMeta(1, "doc1.txt", 15),
        2: DocMeta(2, "doc2.txt", 20),
        3: DocMeta(3, "doc3.txt", 25),
    }
    # term 'python': docs 0, 1, 2
    idx.postings["python"] = [Posting(0, 1), Posting(1, 2), Posting(2, 1)]
    # term 'code': docs 1, 2, 3
    idx.postings["code"] = [Posting(1, 1), Posting(2, 3), Posting(3, 1)]
    # term 'speed': doc 2
    idx.postings["speed"] = [Posting(2, 2)]
    return idx


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_and(mock_index, engine):
    searcher = BooleanSearchEngine(mock_index, engine=engine)
    # python AND code -> docs 1, 2
    res = searcher.search("python code")
    assert res == [1, 2]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_or(mock_index, engine):
    searcher = BooleanSearchEngine(mock_index, engine=engine)
    # python OR speed -> docs 0, 1, 2
    res = searcher.search("python OR speed")
    assert res == [0, 1, 2]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_not(mock_index, engine):
    searcher = BooleanSearchEngine(mock_index, engine=engine)
    # python NOT code -> doc 0
    res = searcher.search("python NOT code")
    assert res == [0]
