import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(r"\b\w+(?:['’`\-]\w+)*\b", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор тексту з NFKC-нормалізацією."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    canonical = normalized.replace("’", "'").replace("`", "'")

    for match in TOKEN_PATTERN.finditer(canonical):
        yield match.group(0)
