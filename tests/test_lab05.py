from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import (
    Index,
    build_parallel_index,
    build_partial,
    collect_corpus_paths,
    merge_partial_indexes,
)


def test_single_partial_merge_matches_lab04_index(sample_corpus: Path) -> None:
    expected = Index.build(sample_corpus, store_positions=True)
    paths = collect_corpus_paths(sample_corpus)
    actual = merge_partial_indexes([build_partial(paths, start_doc_id=0, store_positions=True)])

    assert actual.num_docs == expected.num_docs
    assert actual.avg_doc_length == pytest.approx(expected.avg_doc_length)
    assert dict(actual) == dict(expected)
    assert [actual.get_doc(i) for i in range(actual.num_docs)] == [
        expected.get_doc(i) for i in range(expected.num_docs)
    ]


@pytest.mark.parametrize("executor_kind", ["serial", "threads", "processes"])
def test_parallel_executors_produce_identical_index(
    sample_corpus: Path, executor_kind: str
) -> None:
    expected = Index.build(sample_corpus, store_positions=True)
    idx, stats = build_parallel_index(
        sample_corpus,
        workers=2,
        executor_kind=executor_kind,  # type: ignore[arg-type]
        store_positions=True,
    )
    assert dict(idx) == dict(expected)
    assert stats.num_chunks >= 1
    assert stats.merge_time_sec >= 0.0


def test_worker_failure_propagates_without_hanging(
    sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_build_partial(
        paths: list[Path], start_doc_id: int = 0, store_positions: bool = True
    ) -> None:
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr("findex.index.build_partial", broken_build_partial)
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        build_parallel_index(sample_corpus, workers=2, executor_kind="threads")


def test_cli_parallel_flags(sample_corpus: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "par.bin"
    res = runner.invoke(
        app, ["index", str(sample_corpus), "--out", str(out), "--workers", "2", "--executor", "processes"]
    )
    assert res.exit_code == 0
    assert out.exists()
