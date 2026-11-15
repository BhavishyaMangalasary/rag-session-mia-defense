"""Step 4e: how many answers does the attacker need? (no LLM; reuses Step 2e-4d answers)

For each condition, re-score the attack using only the first q answers per document
(q = 1..5). Under no defense this shows how much leaks before a "wait and see" defense
(like the session defense with K=3) can react.

  python scripts/06_query_budget.py                            -> condition none
  python scripts/06_query_budget.py none mirabel session       -> several conditions

Output: results/query_budget.csv
"""
import json
import sys

import pandas as pd
from tqdm import tqdm

from ragmia import config as C
from ragmia.attack import NLI, doc_text, membership_score, score_answer
from ragmia.metrics import attack_metrics

conditions = sys.argv[1:] or ["none"]
targets = {t["id"]: t for t in map(json.loads, (C.DATA_DIR / "targets.jsonl").open())}
nli = NLI()

rows = []
for cond in conditions:
    records = [json.loads(line) for line in (C.RESULTS_DIR / "raw" / f"attack_{cond}.jsonl").open()]
    judged = {}
    for r in tqdm(records, desc=f"Scoring [{cond}]"):
        document = doc_text(targets[r["id"]])
        judged[r["id"]] = [score_answer(nli, document, t["answer"]) for t in r["turns"]]
    labels = [r["label"] for r in records]
    for q in range(1, C.NUM_QUERIES + 1):
        scores = [membership_score(judged[r["id"]][:q]) for r in records]
        m = attack_metrics(labels, scores)
        rows.append({"condition": cond, "queries": q, "auc": round(m["auc"], 3),
                     "auc_ci_low": round(m["auc_95ci"][0], 3), "auc_ci_high": round(m["auc_95ci"][1], 3),
                     "tpr@1%fpr": round(m["tpr@1%fpr"], 2), "tpr@10%fpr": round(m["tpr@10%fpr"], 2)})

df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / "query_budget.csv", index=False)
print()
print(df.to_string(index=False))