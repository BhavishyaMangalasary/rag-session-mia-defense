"""Step 5: what does each defense cost honest users, and how does the session rule trade off?
(no LLM; replays retrieval only)

Output:
  data/honest_sessions.jsonl         the honest sessions (built once, fixed seed)
  results/honest_sessions.csv        per defense: honest flag rates and utility cost, attack flag rate
  results/session_sweep.csv          session defense for every (window N, threshold K) pair
"""
import json

import pandas as pd

from ragmia import config as C
from ragmia.data import load_beir
from ragmia.rag import Retriever
from ragmia.sessions import attack_flag_rate, build_honest_sessions, honest_summary, replay

_, queries, doc_to_qids = load_beir(C.RAW_DIR / C.DATASET)
split = json.loads((C.DATA_DIR / "split.json").read_text())
qid_to_docs = {}
for d, qids in doc_to_qids.items():
    for q in qids:
        qid_to_docs.setdefault(q, set()).add(d)

path = C.DATA_DIR / "honest_sessions.jsonl"
if not path.exists():
    sessions = build_honest_sessions(queries, doc_to_qids, split["store_ids"])
    path.write_text("".join(json.dumps(s) + "\n" for s in sessions))
sessions = [json.loads(line) for line in path.open()]
attacks = [json.loads(line) for line in (C.DATA_DIR / "attack_queries.jsonl").open()]
print(f"{sum(s['type'] == 'focused' for s in sessions)} focused + "
      f"{sum(s['type'] == 'browsing' for s in sessions)} browsing honest sessions, "
      f"{len(attacks)} attack sessions")

retriever = Retriever()


def session_scores(list_of_query_lists):
    flat = [q for qs in list_of_query_lists for q in qs]
    emb = retriever.model.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    scores = emb @ retriever.emb.T
    out, i = [], 0
    for qs in list_of_query_lists:
        out.append(list(scores[i:i + len(qs)]))
        i += len(qs)
    return out


honest_scores = session_scores([s["queries"] for s in sessions])
attack_scores = session_scores([a["queries"] for a in attacks])
ids = retriever.ids


def get(summary, kind, key):
    return round(summary[kind][key], 3) if kind in summary else float("nan")


# ---- 1. Every defense at its default setting ----
rows = []
for name in ["none", "mirabel", "session", "mirabel_session"]:
    h = honest_summary(replay(name, honest_scores, ids), sessions, qid_to_docs)
    rows.append({
        "defense": name,
        "attack_sessions_flagged": round(attack_flag_rate(replay(name, attack_scores, ids)), 3),
        "focused_sessions_flagged": get(h, "focused", "flagged"),
        "browsing_sessions_flagged": get(h, "browsing", "flagged"),
        "focused_queries_changed": get(h, "focused", "queries_changed"),
        "browsing_queries_changed": get(h, "browsing", "queries_changed"),
        "focused_relevant_lost": get(h, "focused", "relevant_lost"),
        "browsing_relevant_lost": get(h, "browsing", "relevant_lost"),
    })
df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / "honest_sessions.csv", index=False)
print("\nDefault settings:")
print(df.set_index("defense").T.to_string())

# ---- 2. Session rule: every window N and threshold K ----
sweep = []
for window in range(2, C.SESSION_LEN + 1):
    for k in range(2, window + 1):
        h = honest_summary(replay("session", honest_scores, ids, window=window, k=k), sessions, qid_to_docs)
        sweep.append({
            "window": window, "k": k,
            "attack_sessions_flagged": round(attack_flag_rate(
                replay("session", attack_scores, ids, window=window, k=k)), 3),
            "focused_sessions_flagged": get(h, "focused", "flagged"),
            "browsing_sessions_flagged": get(h, "browsing", "flagged"),
        })
sw = pd.DataFrame(sweep)
sw.to_csv(C.RESULTS_DIR / "session_sweep.csv", index=False)
print("\nSession rule sweep (window N, threshold K):")
print(sw.to_string(index=False))