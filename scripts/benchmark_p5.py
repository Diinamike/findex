from pathlib import Path
from findex.index import open_index
from findex.scoring import BM25, TfIdf
from findex.search import SearchEngine

# 10 запитів на корпусі Project Gutenberg + очікувані релевантні документи у топ-5
BENCHMARK = [
    ("sherlock holmes", {"The_Adventures_of_Sherlock_Holmes_1661.txt"}),
    ("hamlet ophelia", {"Hamlet_1787.txt"}),
    ("macbeth duncan", {"Macbeth_1795.txt"}),
    ("othello desdemona", {"Othello_1793.txt"}),
    ("romeo juliet", {"Romeo_and_Juliet_1777.txt"}),
    ("darwin natural selection", {"The_Autobiography_of_Charles_Darwin_2010.txt", "Darwin_and_Modern_Science_1909.txt"}),
    ("odysseus penelope", {"The_Odyssey_1727.txt"}),
    ("julius caesar brutus", {"Julius_Caesar_1785.txt"}),
    ("don quixote sancho", {"Don_Quijote_2000.txt"}),
    ("clausewitz war strategy", {"On_War_1946.txt"}),
]

with open_index(Path("index_lab03.bin")) as idx:
    tfidf = TfIdf()
    bm25 = BM25(k1=1.5, b=0.75)
    engine = SearchEngine(idx, default_scorer=bm25)

    print(f"| {'#':<2} | {'Query':<26} | {'Expected in Top-5':<44} | {'TF-IDF P@5':<10} | {'BM25 P@5':<10} |")
    print(f"|{'-'*4}|{'-'*28}|{'-'*46}|{'-'*12}|{'-'*12}|")

    tfidf_scores = []
    bm25_scores = []

    for i, (q, expected) in enumerate(BENCHMARK, 1):
        res_tfidf = [r.doc_name for r in engine.search(q, top_k=5, scorer=tfidf, with_snippets=False)]
        res_bm25 = [r.doc_name for r in engine.search(q, top_k=5, scorer=bm25, with_snippets=False)]

        p_tfidf = 1.0 if any(doc in expected for doc in res_tfidf) else 0.0
        p_bm25 = 1.0 if any(doc in expected for doc in res_bm25) else 0.0
        tfidf_scores.append(p_tfidf)
        bm25_scores.append(p_bm25)

        exp_str = ", ".join(sorted(expected))
        print(f"| {i:<2} | {q:<26} | {exp_str:<44} | {p_tfidf:<10.2f} | {p_bm25:<10.2f} |")

    print(f"| **Avg** | **10 queries** | — | **{sum(tfidf_scores)/len(tfidf_scores):.2f}** | **{sum(bm25_scores)/len(bm25_scores):.2f}** |")
