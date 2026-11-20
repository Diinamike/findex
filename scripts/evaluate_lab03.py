from pathlib import Path
from findex.index import Index, open_index
from findex.scoring import BM25, TfIdf
from findex.search import SearchEngine

INDEX_PATH = Path("index_lab03.bin")
CORPUS_DIR = Path("data/corpus")

print("=== 1. BUILD & SAVE INDEX (@timed) ===")
idx = Index.build(CORPUS_DIR, store_positions=True)
idx.save(INDEX_PATH)
print(f"Saved: {idx!r}, avg_doc_length={idx.avg_doc_length:.1f}\n")

print("=== 2. OPEN_INDEX + COMPLEX QUERY + CACHE HIT + SNIPPETS ===")
with open_index(INDEX_PATH) as loaded_idx:
    bm25 = BM25(k1=1.5, b=0.75)
    tfidf = TfIdf()
    engine = SearchEngine(loaded_idx, default_scorer=bm25)

    complex_q = '(sherlock OR holmes) AND "baker street" NOT watson'
    print(f"\nQuery: {complex_q}")
    res_first = engine.search(complex_q, top_k=3)
    for r in res_first:
        print(f"  [{r.score:.4f}] {r.doc_name}: {r.snippet}")

    print("\nRepeat same query (testing @lru_cache):")
    _ = engine.search(complex_q, top_k=3)
    print(f"Cache stats: {engine.cache_info}\n")

    print("=== 3. THREE RANKING SANITY CHECKS ===")
    # Check 1: рідкий термін вище частого (за однакового tf=1 та середньої довжини)
    avg_len = int(loaded_idx.avg_doc_length)
    s_rare = bm25.score_term("sherlock", tf=1, doc_length=avg_len, index=loaded_idx)
    s_freq = bm25.score_term("time", tf=1, doc_length=avg_len, index=loaded_idx)
    print(f"Check 1 (Rare 'sherlock' vs Frequent 'time', tf=1): {s_rare:.4f} > {s_freq:.4f} -> {s_rare > s_freq}")

    # Check 2: 20-те повторення майже нічого не додає (насичення TF у BM25)
    s_tf1 = bm25.score_term("sherlock", tf=1, doc_length=avg_len, index=loaded_idx)
    s_tf19 = bm25.score_term("sherlock", tf=19, doc_length=avg_len, index=loaded_idx)
    s_tf20 = bm25.score_term("sherlock", tf=20, doc_length=avg_len, index=loaded_idx)
    gain_first = s_tf1
    gain_20th = s_tf20 - s_tf19
    print(f"Check 2 (Saturation): 1st occ +{gain_first:.4f}, 20th occ +{gain_20th:.6f} -> {gain_20th < gain_first * 0.02}")

    # Check 3: короткий документ з 1 входженням вище дуже довгого з 1 входженням
    s_short = bm25.score_term("sherlock", tf=1, doc_length=1000, index=loaded_idx)
    s_long = bm25.score_term("sherlock", tf=1, doc_length=200000, index=loaded_idx)
    print(f"Check 3 (Length norm, tf=1): short(1k)={s_short:.4f} > long(200k)={s_long:.4f} -> {s_short > s_long}\n")
