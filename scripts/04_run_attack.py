"""Step 2e: run the MEntA attack against the undefended RAG (condition A) and score it.

Outputs:
  results/raw/attack_none.jsonl     every query, answer and (server-side) retrieved docs
  results/attack_none_scores.csv    per document: hits, refusals, membership score
  results/attack_none_metrics.json  AUC (+95% CI), TPR at 1% and 10% FPR
Safe to stop and re-run: documents already queried are skipped.
"""
import json

import pandas as pd
from tqdm import tqdm

from ragmia import config as C
from ragmia.attack import NLI, doc_text, membership_score, score_answer
from ragmia.data import load_beir
from ragmia.metrics import attack_metrics
from ragmia.rag import RAG

CONDITION = "none"

raw_dir = C.RESULTS_DIR / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
raw_path = raw_dir / f"attack_{CONDITION}.jsonl"

# ---- Phase 1: send each target's 5 queries to the RAG and record the answers ----
attacks = [json.loads(line) for line in (C.DATA_DIR / "attack_queries.jsonl").open()]
done = {json.loads(line)["id"] for line in raw_path.open()} if raw_path.exists() else set()
todo = [a for a in attacks if a["id"] not in done]
print(f"Querying: {len(done)} documents already done, {len(todo)} to go")

if todo:
    corpus, _, _ = load_beir(C.RAW_DIR / C.DATASET)
    rag = RAG(corpus)
    with raw_path.open("a") as f:
        for a in tqdm(todo, desc="Querying RAG"):
            turns = []
            for q in a["queries"]:
                out = rag.answer(q)
                turns.append({"query": q, "answer": out["answer"], "retrieved": out["retrieved"]})
            f.write(json.dumps({"id": a["id"], "label": a["label"], "turns": turns}) + "\n")
            f.flush()

# ---- Phase 2: fact-check every answer against the attacker's copy of the document ----
records = [json.loads(line) for line in raw_path.open()]
targets = {t["id"]: t for t in map(json.loads, (C.DATA_DIR / "targets.jsonl").open())}
nli = NLI()
rows = []
for r in tqdm(records, desc="Scoring answers"):
    document = doc_text(targets[r["id"]])
    judged = [score_answer(nli, document, t["answer"]) for t in r["turns"]]
    rows.append({
        "id": r["id"],
        "label": r["label"],
        "hits": sum(h for h, _ in judged),
        "refusals": sum(i for _, i in judged),
        "target_retrieved": sum(r["id"] in [d for d, _ in t["retrieved"]] for t in r["turns"]),
        "score": membership_score(judged),
    })
df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / f"attack_{CONDITION}_scores.csv", index=False)

# ---- Phase 3: how well does the score separate members from non-members? ----
metrics = attack_metrics(df["label"], df["score"])
(C.RESULTS_DIR / f"attack_{CONDITION}_metrics.json").write_text(json.dumps(metrics, indent=1))

summary = df.groupby(df["label"].map({1: "member", 0: "non-member"}))[
    ["hits", "refusals", "target_retrieved", "score"]].mean().round(2)
print("\nAverage per document (out of 5 queries):")
print(summary.to_string())
print(f"\nAUC        : {metrics['auc']:.3f}  (95% CI {metrics['auc_95ci'][0]:.3f}-{metrics['auc_95ci'][1]:.3f})")
print(f"TPR@1%FPR  : {metrics['tpr@1%fpr']:.2f}")
print(f"TPR@10%FPR : {metrics['tpr@10%fpr']:.2f}")