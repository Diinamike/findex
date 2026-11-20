import pytest

from findex.tokenizer import tokenize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ﬁle ﬂow", ["file", "flow"]),  # Unicode NFKC ligatures
        ("Café naïve résumé", ["café", "naïve", "résumé"]),  # Unicode diacritics
        ("don't об’єкт м`ясо", ["don't", "об'єкт", "м'ясо"]),  # Apostrophes
        ("клієнт-серверний high-tech — тест", ["клієнт-серверний", "high-tech", "тест"]),
        ("Версія 3.12 у 2026 році: $100!", ["версія", "3", "12", "у", "2026", "році", "100"]),
        ("   \n\t  ... --- !!!  ", []),  # Empty / punctuation only
        ("Київ-Пасажирський, Україна!", ["київ-пасажирський", "україна"]),
    ],
)
def test_tokenize_parametrized(raw: str, expected: list[str]) -> None:
    assert list(tokenize(raw)) == expected
