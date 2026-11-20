import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.index import DocumentMeta, Index, merge_postings, open_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse_query
from findex.scoring import BM25, Scorer, SearchResult, TfIdf
from findex.search import SearchEngine, build_snippet


def test_index_mapping_and_properties(sample_index: Index) -> None:
    assert sample_index.num_docs == 4
    assert "sherlock" in sample_index
    assert "nonexistent" not in sample_index
    assert len(sample_index) == len(list(iter(sample_index)))
    assert sample_index.df("sherlock") == 3
    assert sample_index.df("missing") == 0
    assert sample_index.avg_doc_length > 0
    assert "status='open'" in repr(sample_index)


def test_open_index_closes_on_exception(sample_index: Index, tmp_path: Path) -> None:
    idx_file = tmp_path / "test.bin"
    sample_index.save(idx_file)

    with pytest.raises(RuntimeError), open_index(idx_file) as idx:
        assert idx.num_docs == 4
        raise RuntimeError("boom")

    assert "status='closed'" in repr(idx)
    with pytest.raises(ValueError):
        _ = idx["sherlock"]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 3, 5, 7], [2, 3, 5, 8], [3, 5]),
        ([], [1, 2, 3], []),
        ([1, 2], [3, 4], []),
        ([0, 4, 9], [0, 4, 9], [0, 4, 9]),
    ],
)
def test_merge_postings(left: list[int], right: list[int], expected: list[int]) -> None:
    assert merge_postings(left, right) == expected


@pytest.mark.parametrize(
    ("query_str", "expected_ast"),
    [
        ("python", Term("python")),
        ('"baker street"', Phrase(("baker", "street"))),
        ("python AND java", And(Term("python"), Term("java"))),
        ("python java", And(Term("python"), Term("java"))),
        ("async OR await", Or(Term("async"), Term("await"))),
        ("NOT java", Not(Term("java"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(
                    And(Term("python"), Or(Term("async"), Term("await"))),
                    Not(Term("java")),
                ),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_query_parser_ast_trees(query_str: str, expected_ast: object) -> None:
    assert parse_query(query_str) == expected_ast


@pytest.mark.parametrize("bad_query", ["", "   ", "(python AND java", '"unclosed phrase', "AND python"])
def test_query_parser_errors(bad_query: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse_query(bad_query)


def test_scorers_protocol_and_sanity_checks(sample_index: Index) -> None:
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()
    assert isinstance(bm25, Scorer)
    assert isinstance(tfidf, Scorer)

    # 1. Рідкий термін ('rare') вище частого ('time') за однакового tf=1
    assert bm25.score_term("rare", 1, 50, sample_index) > bm25.score_term("time", 1, 50, sample_index)

    # 2. Насичення частоти (20-те входження додає значно менше за 1-ше)
    gain_1 = bm25.score_term("sherlock", 1, 50, sample_index)
    gain_20 = bm25.score_term("sherlock", 20, 50, sample_index) - bm25.score_term(
        "sherlock", 19, 50, sample_index
    )
    assert gain_20 < gain_1 * 0.05

    # 3. Короткий документ з tf=1 вище дуже довгого документа з tf=1
    short_score = bm25.score_term("sherlock", 1, 3, sample_index)
    long_score = bm25.score_term("sherlock", 1, 251, sample_index)
    assert short_score > long_score


def test_search_engine_ranking_cache_and_snippets(sample_index: Index) -> None:
    engine = SearchEngine(sample_index, default_scorer=BM25())
    res1 = engine.search('sherlock AND "baker street"', top_k=3, engine="merge")
    assert len(res1) == 1
    assert res1[0].doc_name == "doc_sherlock.txt"
    assert "[Sherlock]" in res1[0].snippet

    res2 = engine.search('sherlock AND "baker street"', top_k=3, scorer=TfIdf(), engine="merge")
    assert res2[0].doc_name == "doc_sherlock.txt"
    assert engine.cache_info.hits == 1
    assert SearchResult(score=2.0, doc_id=1, doc_name="b") > SearchResult(
        score=1.0, doc_id=2, doc_name="a"
    )
    assert "[Python]" in build_snippet("Hello Python world", ["python"], window=20)


def test_save_and_load_without_positions(sample_corpus: Path, tmp_path: Path) -> None:
    idx = Index.build(sample_corpus, store_positions=False)
    out = tmp_path / "nopos.bin"
    idx.save(out)
    loaded = Index.load(out)
    assert loaded.has_positions is False
    assert dict(loaded) == dict(idx)
    with pytest.raises(ValueError):
        parse_query('"baker street"').evaluate(loaded)


def test_cli_workflow_and_error_handling(sample_corpus: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    idx_file = tmp_path / "cli_idx.bin"

    # 1. findex index
    res_idx = runner.invoke(app, ["-v", "index", str(sample_corpus), "--out", str(idx_file)])
    assert res_idx.exit_code == 0
    assert idx_file.exists()

    # 2. findex search (table + --json)
    res_table = runner.invoke(app, ["search", str(idx_file), "sherlock", "--top-k", "2"])
    assert res_table.exit_code == 0
    assert "doc_short.txt" in res_table.stdout

    res_json = runner.invoke(app, ["search", str(idx_file), "python", "--json"])
    assert res_json.exit_code == 0
    assert '"doc_name": "doc_python.txt"' in res_json.stdout

    # 3. findex stats
    res_stats = runner.invoke(app, ["stats", str(idx_file), "--top", "5"])
    assert res_stats.exit_code == 0
    assert "Unique terms" in res_stats.stdout

    # 4. Очікувані помилки -> exit code 1 без необробленого винятку
    err_missing = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert err_missing.exit_code == 1
    assert err_missing.exception is None or isinstance(err_missing.exception, SystemExit)

    err_syntax = runner.invoke(app, ["search", str(idx_file), "(unclosed"])
    assert err_syntax.exit_code == 1


@given(
    left=st.lists(st.integers(min_value=0, max_value=500), max_size=50).map(lambda x: sorted(set(x))),
    right=st.lists(st.integers(min_value=0, max_value=500), max_size=50).map(lambda x: sorted(set(x))),
)
def test_hypothesis_merge_equals_set_intersection(left: list[int], right: list[int]) -> None:
    assert merge_postings(left, right) == sorted(set(left) & set(right))


@given(
    docs_texts=st.lists(
        st.text(alphabet="abcde xyz ", min_size=1, max_size=80),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=30)
def test_hypothesis_postings_always_sorted(docs_texts: list[str], tmp_path_factory: pytest.TempPathFactory) -> None:
    d = tmp_path_factory.mktemp("hyp_sorted")
    for i, txt in enumerate(docs_texts):
        (d / f"d{i}.txt").write_text(txt, encoding="utf-8")
    idx = Index.build(d, store_positions=True)
    for term in idx:
        doc_ids = [doc_id for doc_id, _, _ in idx[term]]
        assert doc_ids == sorted(doc_ids)
        assert len(doc_ids) == len(set(doc_ids))


@given(
    docs_texts=st.lists(
        st.text(alphabet="alpha beta gamma ", min_size=1, max_size=60),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=25)
def test_hypothesis_save_load_roundtrip(docs_texts: list[str], tmp_path_factory: pytest.TempPathFactory) -> None:
    d = tmp_path_factory.mktemp("hyp_rt")
    for i, txt in enumerate(docs_texts):
        (d / f"doc_{i}.txt").write_text(txt, encoding="utf-8")
    idx = Index.build(d, store_positions=True)
    bin_path = d / "rt.bin"
    idx.save(bin_path)
    loaded = Index.load(bin_path)

    assert loaded.num_docs == idx.num_docs
    assert dict(loaded) == dict(idx)
    assert loaded.avg_doc_length == pytest.approx(idx.avg_doc_length)
    assert [DocumentMeta(i, loaded.get_doc(i).name, loaded.get_doc(i).path, loaded.get_doc(i).length) for i in range(loaded.num_docs)] == [
        idx.get_doc(i) for i in range(idx.num_docs)
    ]


@pytest.mark.slow
def test_slow_large_synthetic_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "large"
    corpus.mkdir()
    for i in range(60):
        (corpus / f"book_{i}.txt").write_text(("sherlock holmes watson london " * 400), encoding="utf-8")
    t0 = time.perf_counter()
    idx = Index.build(corpus, store_positions=True)
    out = tmp_path / "large.bin"
    idx.save(out)
    loaded = Index.load(out)
    assert loaded.num_docs == 60
    assert time.perf_counter() - t0 >= 0.0
