"""Step 3d: how often does Mirabel flag honest queries vs attack queries? (detection only, no LLM)

Honest queries: every real user query that ships with the dataset (NFCorpus or SCIDOCS).
Attack queries: the 500 MEntA queries from Step 2c.
Output: results/mirabel_detection.json
"""
import json

import numpy as np

from ragmia import config as C
from ragmia.data import load_beir
from ragmia.defenses import Mirabel
from ragmia.rag import Retriever

retriever = Retriever()
mirabel = Mirabel()


def flagged(queries):
    """True for each query that Mirabel would flag (and hide a document for)."""
    emb = retriever.model.encode(queries, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=True)
    return np.array([bool(mirabel.check(s, retriever.ids)) for s in emb @ retriever.emb.T])


_, queries, _ = load_beir(C.RAW_DIR / C.DATASET)
print(f"Checking {len(queries)} honest {C.DATASET} queries ...")
benign = flagged(list(queries.values()))

attacks = [json.loads(line) for line in (C.DATA_DIR / "attack_queries.jsonl").open()]
print(f"Checking {len(attacks) * C.NUM_QUERIES} attack queries ...")
att = flagged([q for a in attacks for q in a["queries"]]).reshape(len(attacks), C.NUM_QUERIES)
labels = np.array([a["label"] for a in attacks])

results = {
    "honest_queries": int(len(benign)),
    "honest_query_flag_rate": float(benign.mean()),
    "attack_query_flag_rate": float(att.mean()),
    "attack_query_flag_rate_member_targets": float(att[labels == 1].mean()),
    "attack_query_flag_rate_nonmember_targets": float(att[labels == 0].mean()),
    "attack_sessions_with_any_flag": float(att.any(axis=1).mean()),
}
C.RESULTS_DIR.mkdir(exist_ok=True)
(C.RESULTS_DIR / "mirabel_detection.json").write_text(json.dumps(results, indent=1))

PAPER_FPR = {"nfcorpus": "59.6%", "scidocs": "88.1%"}   # MEntA paper, Mirabel's benign FPR
paper_ref = f"  (paper: {PAPER_FPR[C.DATASET]})" if C.DATASET in PAPER_FPR else ""
print(f"\nHonest queries wrongly flagged : {results['honest_query_flag_rate']:.1%}{paper_ref}")
print(f"Attack queries flagged         : {results['attack_query_flag_rate']:.1%}")
print(f"  ... targets in the cabinet   : {results['attack_query_flag_rate_member_targets']:.1%}")
print(f"  ... targets not in cabinet   : {results['attack_query_flag_rate_nonmember_targets']:.1%}")
print(f"Attack sessions with >=1 flag  : {results['attack_sessions_with_any_flag']:.1%}")