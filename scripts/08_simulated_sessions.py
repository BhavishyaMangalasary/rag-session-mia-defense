"""Step 5d: stress-test the defenses with harder honest sessions (Qwen writes the questions).

Two extra session types, one per target document (100 each):
  deep_dive - an honest researcher who knows only the TOPIC (the paper's title) and asks
              5 follow-up questions that dig into it. Harder than real NFCorpus sessions.
  reader    - someone who HAS the document and asks 5 specific questions about it, without
              the attacker's "I am inquiring about ..." prefix. This is what MEntA does, so a
              defense cannot tell it apart from an attack: it marks the upper bound.

Output:
  data/simulated_sessions.jsonl     the deep-dive sessions (written once; safe to re-run)
  results/simulated_sessions.csv    share of sessions each defense flags
"""
import json

import pandas as pd
from tqdm import tqdm

from ragmia import config as C
from ragmia.rag import Retriever
from ragmia.sessions import attack_flag_rate, replay, write_deep_dive

targets = [json.loads(line) for line in (C.DATA_DIR / "targets.jsonl").open()]
attacks = [json.loads(line) for line in (C.DATA_DIR / "attack_queries.jsonl").open()]

path = C.DATA_DIR / "simulated_sessions.jsonl"
done = {json.loads(line)["about"] for line in path.open()} if path.exists() else set()
todo = [t for t in targets if t["id"] not in done]
print(f"Deep-dive sessions: {len(done)} done, {len(todo)} to go (query model: {C.QUERY_MODEL})")
with path.open("a") as f:
    for t in tqdm(todo, desc="Writing deep-dive sessions"):
        f.write(json.dumps({"type": "deep_dive", "about": t["id"],
                            "queries": write_deep_dive(t["title"])}) + "\n")
        f.flush()

deep_dive = [json.loads(line) for line in path.open()]
reader = [{"type": "reader", "about": a["id"], "queries": a["questions"]} for a in attacks]
example = deep_dive[0]
print(f"\nExample deep-dive session (topic of {example['about']}):")
for q in example["queries"]:
    print(" -", q)

retriever = Retriever()


def session_scores(list_of_query_lists):
    flat = [q for qs in list_of_query_lists for q in qs]
    emb = retriever.model.encode(flat, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    scores = emb @ retriever.emb.T
    out, i = [], 0
    for qs in list_of_query_lists:
        out.append(list(scores[i:i + len(qs)]))
        i += len(qs)
    return out


sets = {
    "attack (MEntA)": session_scores([a["queries"] for a in attacks]),
    "honest deep-dive": session_scores([s["queries"] for s in deep_dive]),
    "reader (= attack without prefix)": session_scores([s["queries"] for s in reader]),
}
rows = []
for name in ["mirabel", "session", "mirabel_session"]:
    row = {"defense": name}
    for label, scores in sets.items():
        row[label] = round(attack_flag_rate(replay(name, scores, retriever.ids)), 3)
    rows.append(row)
for window, k in [(3, 2), (5, 2)]:
    row = {"defense": f"session N={window} K={k}"}
    for label, scores in sets.items():
        row[label] = round(attack_flag_rate(replay("session", scores, retriever.ids, window=window, k=k)), 3)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / "simulated_sessions.csv", index=False)
print("\nShare of sessions flagged:")
print(df.to_string(index=False))