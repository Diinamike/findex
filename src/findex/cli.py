from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from findex.index import Index, open_index
from findex.query import EngineName, QuerySyntaxError
from findex.scoring import ScorerName, get_scorer
from findex.search import SearchEngine

app = typer.Typer(
    name="findex",
    help="Full-text search engine CLI (inverted index, boolean/phrase queries, TF-IDF & BM25).",
    no_args_is_help=True,
)

err_console = Console(stderr=True)
out_console = Console()


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logger = logging.getLogger("findex")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase diagnostic logging verbosity (-v for INFO, -vv for DEBUG).",
        ),
    ] = 0,
) -> None:
    _configure_logging(verbose)


@app.command("index")
def build_index(
    corpus_dir: Annotated[Path, typer.Argument(help="Directory containing .txt documents.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output binary index path.")] = Path(
        "index.bin"
    ),
    positions: Annotated[
        bool,
        typer.Option("--positions/--no-positions", help="Store token positions for phrase search."),
    ] = True,
) -> None:
    """Build an inverted index from a directory of text files."""
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        err_console.print(f"[red]Error:[/red] Corpus directory not found: {corpus_dir}")
        raise typer.Exit(code=1)

    total_files = sum(
        1 for p in corpus_dir.rglob("*.txt") if p.is_file() and not p.name.startswith(".")
    )
    if total_files == 0:
        err_console.print(f"[red]Error:[/red] No .txt files found in {corpus_dir}")
        raise typer.Exit(code=1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing documents...", total=total_files)

            def on_progress(count: int, filename: str) -> None:
                progress.update(task, completed=count, description=f"Indexing {filename[:28]}")

            idx = Index.build(corpus_dir, store_positions=positions, on_progress=on_progress)
            idx.save(out)
            summary = repr(idx)
            idx.close()
    except OSError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    err_console.print(f"[green]Saved[/green] {summary} -> {out}")


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to binary index file.")],
    query: Annotated[str, typer.Argument(help="Boolean or phrase query.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of results to show.")] = 5,
    scorer: Annotated[
        ScorerName, typer.Option("--scorer", "-s", help="Ranking model: bm25 or tfidf.")
    ] = "bm25",
    engine: Annotated[
        EngineName, typer.Option("--engine", "-e", help="Boolean intersection engine: set or merge.")
    ] = "set",
    as_json: Annotated[
        bool, typer.Option("--json", help="Output clean JSON results to stdout.")
    ] = False,
) -> None:
    """Search the index and display ranked results."""
    try:
        with open_index(index_path) as idx:
            search_engine = SearchEngine(idx, default_scorer=get_scorer(scorer))
            results = search_engine.search(query, top_k=top_k, engine=engine, with_snippets=True)
    except (FileNotFoundError, QuerySyntaxError, ValueError, OSError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if as_json:
        payload = [
            {
                "rank": rank,
                "doc_id": item.doc_id,
                "doc_name": item.doc_name,
                "score": round(item.score, 6),
                "snippet": item.snippet,
            }
            for rank, item in enumerate(results, start=1)
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    if not results:
        err_console.print("No matching documents found.")
        return

    table = Table(title=f"Search results for: {query} ({scorer.upper()})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Document", style="bold")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Snippet")

    for rank, item in enumerate(results, start=1):
        table.add_row(str(rank), item.doc_name, f"{item.score:.4f}", item.snippet)

    out_console.print(table)


@app.command("stats")
def stats_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to binary index file.")],
    top_n: Annotated[int, typer.Option("--top", "-n", help="Top frequent terms to show.")] = 10,
) -> None:
    """Display statistics for a saved inverted index."""
    try:
        with open_index(index_path) as idx:
            num_docs = idx.num_docs
            num_terms = len(idx)
            avg_len = idx.avg_doc_length
            has_pos = idx.has_positions

            term_freqs: list[tuple[int, str]] = []
            for term in idx:
                total_tf = sum(tf for _, tf, _ in idx[term])
                term_freqs.append((total_tf, term))
            term_freqs.sort(reverse=True)
            top_terms = term_freqs[:top_n]
    except (FileNotFoundError, ValueError, OSError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    summary = Table(title=f"Index Statistics: {index_path}")
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Documents", str(num_docs))
    summary.add_row("Unique terms", str(num_terms))
    summary.add_row("Avg document length (tokens)", f"{avg_len:.1f}")
    summary.add_row("Positional index", "Yes" if has_pos else "No")
    out_console.print(summary)

    if top_terms:
        freq_table = Table(title=f"Top {len(top_terms)} Terms")
        freq_table.add_column("#", justify="right", style="cyan")
        freq_table.add_column("Term", style="bold")
        freq_table.add_column("Total occurrences", justify="right", style="green")
        for i, (tf_sum, term) in enumerate(top_terms, start=1):
            freq_table.add_row(str(i), term, str(tf_sum))
        out_console.print(freq_table)
