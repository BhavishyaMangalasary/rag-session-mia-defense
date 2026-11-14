"""Step 1c: download NFCorpus, embed it, and build the member / non-member split.

Run once:  python scripts/01_prepare_data.py
Outputs (in data/):
  corpus_ids.json   - doc ids in embedding order
  corpus_emb.npy    - normalized embeddings of every doc (title + text)
  split.json        - members, nonmembers, store_ids (store = corpus minus non-members)
  targets.jsonl     - the 100 target docs with label 1 (member) / 0 (non-member)
  doc_queries.json  - real NFCorpus user queries per target doc (for benign sessions later)
"""
import csv
import io
import json
import random
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from ragmia import config as C


def download():
    folder = C.RAW_DIR / C.DATASET
    if folder.exists():
        return folder
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {C.BEIR_URL} ...")
    with urllib.request.urlopen(C.BEIR_URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(C.RAW_DIR)
    return folder


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(folder):
    corpus = {d["_id"]: {"title": d.get("title", ""), "text": d["text"]}
              for d in read_jsonl(folder / "corpus.jsonl")}
    queries = {q["_id"]: q["text"] for q in read_jsonl(folder / "queries.jsonl")}
    doc_to_qids = defaultdict(set)
    for tsv in (folder / "qrels").glob("*.tsv"):
        with open(tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if int(row["score"]) > 0:
                    doc_to_qids[row["corpus-id"]].add(row["query-id"])
    return corpus, queries, doc_to_qids


def doc_string(doc):
    return f"{doc['title']}. {doc['text']}".strip(". ")


def embed_corpus(corpus):
    ids = list(corpus)
    emb_path, ids_path = C.DATA_DIR / "corpus_emb.npy", C.DATA_DIR / "corpus_ids.json"
    if emb_path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        print("Using cached embeddings.")
        return ids, np.load(emb_path)
    model = SentenceTransformer(C.RETRIEVER_MODEL)
    print(f"Embedding {len(ids)} docs on {model.device} ...")
    emb = model.encode([doc_string(corpus[i]) for i in ids], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(ids))
    return ids, emb


def make_split(corpus, doc_to_qids, ids, emb):
    idx = {d: i for i, d in enumerate(ids)}
    pool = [d for d in ids
            if len(corpus[d]["text"].split()) >= C.MIN_WORDS
            and len(doc_to_qids.get(d, ())) >= C.MIN_QUERIES_PER_DOC]
    print(f"Candidate pool: {len(pool)} docs")
    random.Random(C.SEED).shuffle(pool)

    # Pick non-members first. Each must have no near-duplicate left in the store,
    # otherwise the RAG could still answer from the duplicate and the label would be wrong.
    nonmembers, removed = [], np.zeros(len(ids), dtype=bool)
    rejected = 0
    for d in pool:
        if len(nonmembers) == C.N_NONMEMBERS:
            break
        sims = emb @ emb[idx[d]]
        sims[idx[d]] = -1
        sims[removed] = -1
        if sims.max() >= C.DUP_SIM_THRESHOLD:
            rejected += 1
            continue
        nonmembers.append(d)
        removed[idx[d]] = True
    print(f"Rejected {rejected} near-duplicate non-member candidates")

    chosen = set(nonmembers)
    members = [d for d in pool if d not in chosen][: C.N_MEMBERS]
    if len(members) < C.N_MEMBERS or len(nonmembers) < C.N_NONMEMBERS:
        raise SystemExit("Pool too small: lower MIN_QUERIES_PER_DOC or MIN_WORDS in config.py")

    store_ids = [d for d in ids if d not in chosen]
    return members, nonmembers, store_ids


def prepare_data():
    C.DATA_DIR.mkdir(exist_ok=True)
    corpus, queries, doc_to_qids = load_beir(download())
    print(f"Corpus: {len(corpus)} docs, {len(queries)} queries")
    ids, emb = embed_corpus(corpus)
    members, nonmembers, store_ids = make_split(corpus, doc_to_qids, ids, emb)

    (C.DATA_DIR / "split.json").write_text(json.dumps(
        {"members": members, "nonmembers": nonmembers, "store_ids": store_ids}, indent=1))
    with open(C.DATA_DIR / "targets.jsonl", "w") as f:
        for label, group in ((1, members), (0, nonmembers)):
            for d in group:
                f.write(json.dumps({"id": d, "label": label, **corpus[d]}) + "\n")
    (C.DATA_DIR / "doc_queries.json").write_text(json.dumps(
        {d: sorted(queries[q] for q in doc_to_qids[d] if q in queries)
         for d in members + nonmembers}, indent=1))

    print(f"Members: {len(members)} | Non-members: {len(nonmembers)} | Store size: {len(store_ids)}")