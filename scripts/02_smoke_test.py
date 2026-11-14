"""Step 1d: check the RAG works before building the attack.

  python scripts/02_smoke_test.py            -> retrieval check only (fast, no LLM)
  python scripts/02_smoke_test.py --generate -> also asks Llama one question about a member and a non-member
"""
import json
import sys

from ragmia import config as C
from ragmia.data import load_beir
from ragmia.rag import RAG

corpus, _, _ = load_beir(C.RAW_DIR / C.DATASET)
split = json.loads((C.DATA_DIR / "split.json").read_text())
rag = RAG(corpus)

# 1. Retrieval check: searching a member's own title should usually return that member.
hits = sum(m in [d for d, _ in rag.retriever.search(corpus[m]["title"])] for m in split["members"])
print(f"Member found in top-{C.TOP_K} when searching its own title: {hits}/{len(split['members'])}")

# 2. Non-members must never be retrievable.
store = set(split["store_ids"])
assert not any(n in store for n in split["nonmembers"]), "A non-member leaked into the store!"
print("No non-member is in the store: OK")

# 3. End-to-end generation (needs Ollama running).
if "--generate" in sys.argv:
    for label in ("members", "nonmembers"):
        d = split[label][0]
        q = f"What does research say about {corpus[d]['title'].lower()}?"
        out = rag.answer(q)
        print(f"\n=== {label[:-1]} {d}\nQ: {q}")
        print("Retrieved:", [(i, round(s, 3)) for i, s in out["retrieved"]])
        print("Target in context:", d in [i for i, _ in out["retrieved"]])
        print("A:", out["answer"][:600])