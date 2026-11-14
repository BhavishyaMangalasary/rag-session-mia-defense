"""Step 1d: the target RAG system (retriever + Ollama generator).

The attacker only ever sees `answer`. The retrieved ids and similarity scores are
server-side information; the defenses (Mirabel, session tracker) will use them.
`exclude` lets a defense hide documents from the context (Mirabel's 'hide' action).
"""
import json

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from ragmia import config as C

SYSTEM_PROMPT = ("You are a helpful assistant. Use the provided context documents "
                 "to answer the user's question.")


class Retriever:
    def __init__(self):
        all_ids = json.loads((C.DATA_DIR / "corpus_ids.json").read_text())
        all_emb = np.load(C.DATA_DIR / "corpus_emb.npy")
        store = set(json.loads((C.DATA_DIR / "split.json").read_text())["store_ids"])
        keep = [i for i, d in enumerate(all_ids) if d in store]
        self.ids = [all_ids[i] for i in keep]
        self.emb = all_emb[keep]
        self.model = SentenceTransformer(C.RETRIEVER_MODEL)

    def search(self, query, k=C.TOP_K, exclude=()):
        q = self.model.encode(query, normalize_embeddings=True)
        scores = self.emb @ q
        out = []
        for i in np.argsort(-scores):
            if self.ids[i] in exclude:
                continue
            out.append((self.ids[i], float(scores[i])))
            if len(out) == k:
                break
        return out


def ollama_chat(messages, model=C.GENERATOR_MODEL, temperature=C.GEN_TEMPERATURE,
                max_tokens=C.GEN_MAX_TOKENS):
    r = requests.post(C.OLLAMA_URL, timeout=300, json={
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


class RAG:
    def __init__(self, corpus):
        self.corpus = corpus  # {doc_id: {"title", "text"}}
        self.retriever = Retriever()

    def answer(self, query, exclude=()):
        hits = self.retriever.search(query, exclude=exclude)
        context = "\n\n".join(
            f"[Document {n}] {self.corpus[d]['title']}\n{self.corpus[d]['text']}"
            for n, (d, _) in enumerate(hits, 1))
        reply = ollama_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ])
        return {"answer": reply, "retrieved": hits}