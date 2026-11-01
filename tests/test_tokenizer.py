from findex.tokenizer import tokenize


def test_unicode_normalization():
    # Комбіновані символи (лігатури або розкладки)
    raw = "ﬁle"  # лігатура fi
    tokens = list(tokenize(raw))
    assert tokens == ["file"]


def test_apostrophes():
    # Різні типи апострофів: прямий, правий типографський, гравіс
    text = "don't об’єкт м`ясо"
    tokens = list(tokenize(text))
    assert tokens == ["don't", "об'єкт", "м'ясо"]


def test_hyphenated_words():
    # Дефіси всередині слова зберігаються, тире між словами відсікається
    text = "клієнт-серверний підхід — це high-tech рішення"
    tokens = list(tokenize(text))
    assert "клієнт-серверний" in tokens
    assert "high-tech" in tokens
    assert "—" not in tokens


def test_digits_and_punctuation():
    # Цифри зберігаються, пунктуація видаляється
    text = "У 2026 році версія 3.12 коштує $100! Чи не так?"
    tokens = list(tokenize(text))
    assert "2026" in tokens
    assert "3" in tokens and "12" in tokens
    assert "100" in tokens
    assert "!" not in tokens and "$" not in tokens and "?" not in tokens
