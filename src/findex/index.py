from __future__ import annotations

import array
import os
import struct
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from findex.corpus import iter_documents
from findex.tokenizer import tokenize
from findex.utils import timed

Posting = tuple[int, int, tuple[int, ...]]
Int32Array = npt.NDArray[np.int32]
MAGIC = b"FIDX04"
_EMPTY_INT32: Int32Array = np.empty(0, dtype=np.int32)


def merge_postings(left: list[int], right: list[int]) -> list[int]:
    """Двовказівниковий перетин двох відсортованих списків doc_id за O(L1 + L2)."""
    result: list[int] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


@dataclass(frozen=True)
class DocumentMeta:
    doc_id: int
    name: str
    path: str
    length: int


class Index(Mapping[str, list[Posting]]):
    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        docs: list[DocumentMeta] | None = None,
        has_positions: bool = True,
        *,
        posting_doc_ids: dict[str, Int32Array] | None = None,
        posting_tfs: dict[str, Int32Array] | None = None,
        posting_positions: dict[str, tuple[tuple[int, ...], ...]] | None = None,
    ) -> None:
        self._docs: list[DocumentMeta] = docs or []
        self.doc_lengths: Int32Array = np.array(
            [d.length for d in self._docs], dtype=np.int32
        )
        self.has_positions: bool = has_positions
        self._closed: bool = False
        self._text_cache: dict[int, str] = {}

        self._postings_cache: dict[str, list[Posting]] = {}
        self._doc_ids: dict[str, Int32Array] = {}
        self._tfs: dict[str, Int32Array] = {}
        self._positions: dict[str, tuple[tuple[int, ...], ...]] = {}

        if posting_doc_ids is not None and posting_tfs is not None:
            self._doc_ids = posting_doc_ids
            self._tfs = posting_tfs
            self._positions = posting_positions or {}
        elif postings:
            self._postings_cache = dict(postings)
            for term, plist in postings.items():
                if plist:
                    self._doc_ids[term] = np.array([p[0] for p in plist], dtype=np.int32)
                    self._tfs[term] = np.array([p[1] for p in plist], dtype=np.int32)
                    if has_positions:
                        self._positions[term] = tuple(p[2] for p in plist)
                else:
                    self._doc_ids[term] = _EMPTY_INT32
                    self._tfs[term] = _EMPTY_INT32
                    if has_positions:
                        self._positions[term] = ()

    def postings_arrays(self, term: str) -> tuple[Int32Array, Int32Array]:
        self._ensure_open()
        d_ids = self._doc_ids.get(term)
        if d_ids is None:
            return _EMPTY_INT32, _EMPTY_INT32
        return d_ids, self._tfs[term]

    def __getitem__(self, term: str) -> list[Posting]:
        self._ensure_open()
        cached = self._postings_cache.get(term)
        if cached is not None:
            return cached
        d_ids = self._doc_ids[term]
        tfs = self._tfs[term]
        pos_tuple = self._positions.get(term)
        if pos_tuple is not None and len(pos_tuple) == len(d_ids):
            plist = [
                (int(d), int(tf), pos)
                for d, tf, pos in zip(d_ids.tolist(), tfs.tolist(), pos_tuple, strict=True)
            ]
        else:
            plist = [(int(d), int(tf), ()) for d, tf in zip(d_ids.tolist(), tfs.tolist(), strict=True)]
        self._postings_cache[term] = plist
        return plist

    def __iter__(self) -> Iterator[str]:
        self._ensure_open()
        return iter(self._doc_ids)

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._doc_ids)

    def __contains__(self, term: object) -> bool:
        self._ensure_open()
        return term in self._doc_ids

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (
            f"Index(docs={self.num_docs}, terms={len(self._doc_ids)}, "
            f"positions={self.has_positions}, status={status!r})"
        )

    @property
    def num_docs(self) -> int:
        return len(self._docs)

    @cached_property
    def avg_doc_length(self) -> float:
        if self.doc_lengths.size == 0:
            return 0.0
        return float(self.doc_lengths.mean())

    def df(self, term: str) -> int:
        d_ids = self._doc_ids.get(term)
        return int(d_ids.size) if d_ids is not None else 0

    def doc_ids_for(self, term: str) -> list[int]:
        d_ids = self._doc_ids.get(term)
        if d_ids is None or d_ids.size == 0:
            return []
        return d_ids.tolist()

    def get_doc(self, doc_id: int) -> DocumentMeta:
        return self._docs[doc_id]

    def read_doc_text(self, doc_id: int) -> str:
        cached = self._text_cache.get(doc_id)
        if cached is not None:
            return cached
        meta = self._docs[doc_id]
        text = Path(meta.path).read_text(encoding="utf-8", errors="replace")
        self._text_cache[doc_id] = text
        return text

    def close(self) -> None:
        self._postings_cache.clear()
        self._doc_ids.clear()
        self._tfs.clear()
        self._positions.clear()
        self._docs.clear()
        self._text_cache.clear()
        self.doc_lengths = _EMPTY_INT32
        self.__dict__.pop("avg_doc_length", None)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("Operation on closed Index")

    @classmethod
    @timed
    def build(
        cls,
        corpus_dir: str | Path,
        store_positions: bool = True,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> Index:
        corpus_path = Path(corpus_dir)
        postings_map: dict[str, list[Posting]] = defaultdict(list)
        docs_meta: list[DocumentMeta] = []

        for doc_id, (filename, text) in enumerate(iter_documents(corpus_path)):
            if on_progress is not None:
                on_progress(doc_id + 1, filename)

            term_positions: dict[str, list[int]] = defaultdict(list)
            token_count = 0

            for pos, token in enumerate(tokenize(text)):
                token_count += 1
                if store_positions:
                    term_positions[token].append(pos)
                elif token not in term_positions:
                    term_positions[token] = [1]
                else:
                    term_positions[token][0] += 1

            for term, pos_list in term_positions.items():
                if store_positions:
                    postings_map[term].append((doc_id, len(pos_list), tuple(pos_list)))
                else:
                    postings_map[term].append((doc_id, pos_list[0], ()))

            full_path = str((corpus_path / filename).resolve())
            docs_meta.append(
                DocumentMeta(doc_id=doc_id, name=filename, path=full_path, length=token_count)
            )

        return cls(postings=dict(postings_map), docs=docs_meta, has_positions=store_positions)

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        with out_path.open("wb") as f:
            f.write(MAGIC)
            f.write(
                struct.pack("<BII", int(self.has_positions), len(self._docs), len(self._doc_ids))
            )

            for doc in self._docs:
                name_b = doc.name.encode("utf-8")
                path_b = doc.path.encode("utf-8")
                f.write(struct.pack("<H", len(name_b)))
                f.write(name_b)
                f.write(struct.pack("<H", len(path_b)))
                f.write(path_b)
                f.write(struct.pack("<I", doc.length))

            for term, d_ids in self._doc_ids.items():
                tfs = self._tfs[term]
                pos_tuple = self._positions.get(term)
                term_b = term.encode("utf-8")
                f.write(struct.pack("<HI", len(term_b), int(d_ids.size)))
                f.write(term_b)

                for idx_i, (doc_id, tf) in enumerate(zip(d_ids.tolist(), tfs.tolist(), strict=True)):
                    f.write(struct.pack("<II", doc_id, tf))
                    if self.has_positions:
                        positions = pos_tuple[idx_i] if pos_tuple is not None else ()
                        arr = array.array("I", positions)
                        arr.tofile(f)

    @classmethod
    @timed
    def load(cls, path: str | Path) -> Index:
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Index file not found: {in_path}")

        with in_path.open("rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError(f"Invalid index file header in {in_path}")

            has_pos_flag, num_docs, num_terms = struct.unpack("<BII", f.read(9))
            has_positions = bool(has_pos_flag)

            docs: list[DocumentMeta] = []
            for doc_id in range(num_docs):
                (name_len,) = struct.unpack("<H", f.read(2))
                name = f.read(name_len).decode("utf-8")
                (path_len,) = struct.unpack("<H", f.read(2))
                doc_path = f.read(path_len).decode("utf-8")
                (length,) = struct.unpack("<I", f.read(4))
                docs.append(DocumentMeta(doc_id=doc_id, name=name, path=doc_path, length=length))

            posting_doc_ids: dict[str, Int32Array] = {}
            posting_tfs: dict[str, Int32Array] = {}
            posting_positions: dict[str, tuple[tuple[int, ...], ...]] = {}

            for _ in range(num_terms):
                term_len, plist_len = struct.unpack("<HI", f.read(6))
                term = f.read(term_len).decode("utf-8")

                d_arr = np.empty(plist_len, dtype=np.int32)
                tf_arr = np.empty(plist_len, dtype=np.int32)
                pos_list: list[tuple[int, ...]] = []

                for i in range(plist_len):
                    doc_id, tf = struct.unpack("<II", f.read(8))
                    d_arr[i] = doc_id
                    tf_arr[i] = tf
                    if has_positions:
                        arr = array.array("I")
                        arr.fromfile(f, tf)
                        pos_list.append(tuple(arr))

                posting_doc_ids[term] = d_arr
                posting_tfs[term] = tf_arr
                if has_positions:
                    posting_positions[term] = tuple(pos_list)

        return cls(
            docs=docs,
            has_positions=has_positions,
            posting_doc_ids=posting_doc_ids,
            posting_tfs=posting_tfs,
            posting_positions=posting_positions,
        )


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    idx = Index.load(path)
    try:
        yield idx
    finally:
        idx.close()


ExecutorName = Literal["serial", "threads", "processes"]


@dataclass(frozen=True)
class PartialIndex:
    docs: list[DocumentMeta]
    postings: dict[str, list[Posting]]


@dataclass(frozen=True)
class BuildStats:
    merge_time_sec: float
    num_chunks: int


def collect_corpus_paths(corpus_dir: str | Path) -> list[Path]:
    root = Path(corpus_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")
    return sorted(
        (p for p in root.rglob("*.txt") if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )


def chunk_paths(paths: Sequence[Path], num_chunks: int) -> list[tuple[int, list[Path]]]:
    if not paths:
        return []
    n_chunks = max(1, min(num_chunks, len(paths)))
    base_size, remainder = divmod(len(paths), n_chunks)
    chunks: list[tuple[int, list[Path]]] = []
    offset = 0
    for i in range(n_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append((offset, list(paths[offset : offset + size])))
        offset += size
    return chunks


def build_partial(
    paths: Sequence[Path],
    start_doc_id: int = 0,
    store_positions: bool = True,
) -> PartialIndex:
    """Функція рівня модуля для побудови часткового індексу у воркері (потік або процес)."""
    postings_map: dict[str, list[Posting]] = defaultdict(list)
    docs_meta: list[DocumentMeta] = []

    for local_idx, file_path in enumerate(paths):
        doc_id = start_doc_id + local_idx
        text = file_path.read_text(encoding="utf-8", errors="replace")

        term_positions: dict[str, list[int]] = defaultdict(list)
        token_count = 0

        for pos, token in enumerate(tokenize(text)):
            token_count += 1
            if store_positions:
                term_positions[token].append(pos)
            elif token not in term_positions:
                term_positions[token] = [1]
            else:
                term_positions[token][0] += 1

        for term, pos_list in term_positions.items():
            if store_positions:
                postings_map[term].append((doc_id, len(pos_list), tuple(pos_list)))
            else:
                postings_map[term].append((doc_id, pos_list[0], ()))

        docs_meta.append(
            DocumentMeta(
                doc_id=doc_id,
                name=file_path.name,
                path=str(file_path.resolve()),
                length=token_count,
            )
        )

    return PartialIndex(docs=docs_meta, postings=dict(postings_map))


def merge_partial_indexes(
    partials: Sequence[PartialIndex],
    store_positions: bool = True,
) -> Index:
    """Злиття часткових індексів із неперетинними впорядкованими діапазонами doc_id."""
    ordered = sorted(partials, key=lambda p: p.docs[0].doc_id if p.docs else 0)
    merged_docs: list[DocumentMeta] = []
    merged_postings: dict[str, list[Posting]] = defaultdict(list)

    for part in ordered:
        merged_docs.extend(part.docs)
        for term, plist in part.postings.items():
            merged_postings[term].extend(plist)

    return Index(postings=dict(merged_postings), docs=merged_docs, has_positions=store_positions)


def build_parallel_index(
    corpus_dir: str | Path,
    workers: int | None = None,
    executor_kind: ExecutorName = "processes",
    store_positions: bool = True,
    on_progress: Callable[[int, str], None] | None = None,
) -> tuple[Index, BuildStats]:
    src_path = Path(corpus_dir)
    if src_path.is_file() and src_path.suffix == ".jsonl":
        t_seq0 = time.perf_counter()
        idx = Index.build(src_path, store_positions=store_positions, on_progress=on_progress)
        return idx, BuildStats(merge_time_sec=time.perf_counter() - t_seq0, num_chunks=1)
    paths = collect_corpus_paths(src_path)
    if not paths:
        return Index(postings={}, docs=[], has_positions=store_positions), BuildStats(0.0, 0)

    n_workers = workers if workers is not None and workers > 0 else (os.cpu_count() or 1)
    chunks = chunk_paths(paths, 1 if executor_kind == "serial" else n_workers)
    partials: list[PartialIndex] = []
    done_docs = 0

    if executor_kind == "serial" or len(chunks) == 1:
        for start_id, chunk in chunks:
            part = build_partial(chunk, start_doc_id=start_id, store_positions=store_positions)
            partials.append(part)
            done_docs += len(chunk)
            if on_progress is not None:
                on_progress(done_docs, chunk[-1].name)
    else:
        pool_cls = ThreadPoolExecutor if executor_kind == "threads" else ProcessPoolExecutor
        pool = pool_cls(max_workers=len(chunks))
        try:
            future_to_chunk = {
                pool.submit(build_partial, chunk, start_id, store_positions): chunk
                for start_id, chunk in chunks
            }
            for fut in as_completed(future_to_chunk):
                part = fut.result()
                chunk = future_to_chunk[fut]
                partials.append(part)
                done_docs += len(chunk)
                if on_progress is not None:
                    on_progress(done_docs, chunk[-1].name)
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)

    t_merge0 = time.perf_counter()
    merged = merge_partial_indexes(partials, store_positions=store_positions)
    merge_elapsed = time.perf_counter() - t_merge0
    return merged, BuildStats(merge_time_sec=merge_elapsed, num_chunks=len(chunks))
