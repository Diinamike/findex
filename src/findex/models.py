from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    term_frequency: int
    positions: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str
    length: int
