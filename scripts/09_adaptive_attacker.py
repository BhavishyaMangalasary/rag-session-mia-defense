"""Step 5e: an adaptive attacker who tries to slip past the session window (no LLM).

The attacker inserts d decoy questions (random real NFCorpus queries) after each of the 5 MEntA
questions, so the target document comes up less often inside the defense's window. We also try
dropping the "I am inquiring about ..." prefix, which makes the questions less concentrated.

For member targets we count, out of the 5 attack questions:
  target_leaked  - the target document still reached the LLM's context (what the attacker wants)
  target_hidden  - the defense kept the target out
Without any defense the target reaches the context on about 4.7 of 5 questions.

Output: results/adaptive_attacker.csv
"""
import json
import random

import pandas as pd

from ragmia import config as C
from ragmia.data import load_beir
from ragmia.rag import Retriever
from ragmia.sessions import replay

_, queries, _ = load_beir(C.RAW_DIR / C.DATASET)
attacks = [json.loads(line) for line in (C.DATA_DIR / "attack_queries.jsonl").open()]
members = [a for a in attacks if a["label"] == 1]
pool = sorted(queries.values())
retriever = Retriever()

# encode every attack question (with and without prefix) and the decoy pool once
texts = sorted({q for a in members for q in a["queries"] + a["questions"]} | set(pool))
emb = retriever.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
score_of = dict(zip(texts, emb @ retriever.emb.T))

DEFENSES = [("none", {}), ("mirabel", {}), ("session", {}), ("mirabel_session", {}),
            ("session", {"window": 3, "k": 2}), ("session", {"window": 5, "k": 2})]

rows = []
for prefix in (True, False):
    for decoys in range(0, 4):
        rng = random.Random(C.SEED + decoys)
        sessions, attack_turns = [], []
        for a in members:
            seq, is_attack = [], []
            for q in (a["queries"] if prefix else a["questions"]):
                seq.append(q)
                is_attack.append(True)
                for _ in range(decoys):
                    seq.append(rng.choice(pool))
                    is_attack.append(False)
            sessions.append([score_of[q] for q in seq])
            attack_turns.append(is_attack)
        for name, params in DEFENSES:
            replayed = replay(name, sessions, retriever.ids, **params)
            leaked, hidden = [], []
            for a, turns, is_attack in zip(members, replayed, attack_turns):
                att = [t for t, flag in zip(turns, is_attack) if flag]
                leaked.append(sum(a["id"] in defended for _, _, defended in att))
                hidden.append(sum(a["id"] in h for h, _, _ in att))
            label = name if not params else f"session N={params['window']} K={params['k']}"
            rows.append({"prefix": prefix, "decoys_per_question": decoys,
                         "queries_sent": C.NUM_QUERIES * (1 + decoys), "defense": label,
                         "target_leaked": round(sum(leaked) / len(leaked), 2),
                         "target_hidden": round(sum(hidden) / len(hidden), 2)})

df = pd.DataFrame(rows)
df.to_csv(C.RESULTS_DIR / "adaptive_attacker.csv", index=False)
for prefix in (True, False):
    sub = df[df["prefix"] == prefix]
    print(f"\nTarget leaked into the context (of 5 attack questions, member targets) - "
          f"{'with' if prefix else 'WITHOUT'} the 'I am inquiring about' prefix:")
    print(sub.pivot(index="defense", columns="decoys_per_question", values="target_leaked")
             .reindex([d if not p else f"session N={p['window']} K={p['k']}" for d, p in DEFENSES])
             .rename(columns=lambda d: f"{d} decoys ({C.NUM_QUERIES * (1 + d)} queries)")
             .to_string())