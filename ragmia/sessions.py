"""Step 5: honest (benign) sessions built from real NFCorpus user queries, and a fast,
LLM-free evaluation of what each defense does to honest and attack sessions.

Two honest session types, SESSION_LEN real queries each:
  focused  - all queries are relevant to the SAME store document (a user digging into one topic:
             the hard case, because it looks like an attacker circling one document)
  browsing - queries drawn at random (a user moving between topics: the easy case)

Whether a defense flags a query depends only on retrieval scores, not on the LLM's answer,
so every defense can be replayed on every session in seconds.
"""
import random
from collections import defaultdict

import numpy as np

from ragmia import config as C
from ragmia.defenses import make_defense


def build_honest_sessions(queries, doc_to_qids, store_ids, seed=C.SEED):
    rng = random.Random(seed)
    store = set(store_ids)
    eligible = sorted(d for d, qids in doc_to_qids.items()
                      if d in store and len([q for q in qids if q in queries]) >= C.SESSION_LEN)
    sessions = []
    for d in rng.sample(eligible, min(C.N_HONEST_SESSIONS, len(eligible))):
        qids = rng.sample(sorted(q for q in doc_to_qids[d] if q in queries), C.SESSION_LEN)
        sessions.append({"type": "focused", "about": d, "qids": qids})
    all_qids = sorted(queries)
    for _ in range(C.N_HONEST_SESSIONS):
        sessions.append({"type": "browsing", "about": None, "qids": rng.sample(all_qids, C.SESSION_LEN)})
    for s in sessions:
        s["queries"] = [queries[q] for q in s["qids"]]
    return sessions


def replay(defense_name, session_scores, ids, top_k=C.TOP_K, **params):
    """Run one defense over sessions given precomputed similarity scores.

    session_scores: list of sessions, each a list of score vectors (one per query).
    Returns, per session, a list of (hidden set, top-k without defense, top-k with defense).
    """
    out = []
    for scores_list in session_scores:
        defense = make_defense(defense_name, **params)
        turns = []
        for scores in scores_list:
            hidden = defense.check(scores, ids)
            order = np.argsort(-scores)
            plain = [ids[i] for i in order[:top_k]]
            defended = [ids[i] for i in order if ids[i] not in hidden][:top_k]
            turns.append((hidden, plain, defended))
        out.append(turns)
    return out


def honest_summary(replayed, sessions, qid_to_docs):
    """Per honest session type: how often the defense raised a flag, and what it cost the user.

    flagged          - the defense hid at least one document during the session
    queries_changed  - share of queries whose top-k context differs from the undefended one
    relevant_lost    - relevant documents (per NFCorpus judgments) lost from the top-k, per query
    """
    rows = defaultdict(list)
    for turns, s in zip(replayed, sessions):
        flagged = any(bool(hidden) for hidden, _, _ in turns)
        changed = [plain != defended for _, plain, defended in turns]
        lost = [len(qid_to_docs.get(q, set()) & set(plain)) - len(qid_to_docs.get(q, set()) & set(defended))
                for (_, plain, defended), q in zip(turns, s["qids"])]
        rows[s["type"]].append((flagged, np.mean(changed), np.mean(lost)))
    return {t: {"flagged": float(np.mean([r[0] for r in v])),
                "queries_changed": float(np.mean([r[1] for r in v])),
                "relevant_lost": float(np.mean([r[2] for r in v]))}
            for t, v in rows.items()}


def attack_flag_rate(replayed):
    """Share of attack sessions where the defense hid at least one document."""
    return float(np.mean([any(bool(h) for h, _, _ in turns) for turns in replayed]))



# ---------------------------------------------------------------------------
# Simulated honest deep-dive sessions (harder than real NFCorpus sessions)
# ---------------------------------------------------------------------------

DEEP_DIVE_PROMPT = """You are a curious health researcher using a medical research assistant.
You are exploring the topic below but have NOT read any specific paper about it.
Write {n} questions you would ask, one after another, where each question follows up on the
previous answer and digs deeper into the same topic. Ask about findings, numbers, populations
and mechanisms, the way a real researcher narrows down a question.

Topic: {topic}

Output format:
QUERY_1: [first question]
QUERY_2: [second question]
...

Write {n} questions now:"""


def write_deep_dive(topic, n=C.SESSION_LEN):
    """An honest user who knows only the topic (a paper title) and asks n focused follow-ups."""
    from ragmia.attack import parse_queries
    from ragmia.rag import ollama_chat

    questions = []
    for temperature in (0.0, 0.7, 0.7):
        raw = ollama_chat([{"role": "user", "content": DEEP_DIVE_PROMPT.format(n=n, topic=topic)}],
                          model=C.QUERY_MODEL, temperature=temperature, max_tokens=500)
        questions = parse_queries(raw)
        if len(questions) >= n:
            return questions[:n]
    raise RuntimeError(f"Only got {len(questions)} questions for topic: {topic}")