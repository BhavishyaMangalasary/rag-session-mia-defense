"""Defenses that sit inside the RAG server, between retrieval and generation.

Every defense has the same interface:
    check(scores, ids) -> set of doc ids to hide from the LLM for this query
where `scores` are the query's cosine similarities to every document in the store and
`ids` are the matching document ids. One defense object is created per session, so a
defense may remember earlier queries (the session-level defense does; Mirabel does not).
"""
import math
from collections import Counter, deque

import numpy as np

from ragmia import config as C


class NoDefense:
    """Condition A: retrieve and answer normally."""
    name = "none"

    def check(self, scores, ids):
        return set()


class Mirabel:
    """Condition B: Mirabel detect-and-hide (Choi et al., Findings of EMNLP 2025, Algorithm 1).

    Per query, with no memory: if the best-matching document is an extreme outlier compared
    with the query's similarity to all other documents (a Gumbel extreme-value test at
    significance level rho), that one document is hidden and the next-ranked one takes its slot.
    """
    name = "mirabel"

    def __init__(self, rho=C.MIRABEL_RHO):
        self.c = -math.log(-math.log(1 - rho))      # Gumbel critical value (~2.97 for rho=0.05)

    def threshold(self, scores):
        top = int(np.argmax(scores))
        rest = np.delete(scores, top)                # similarities to every other document
        n = len(rest)
        mu, sigma = float(rest.mean()), float(rest.std())
        root = math.sqrt(2 * math.log(n))
        tau = mu + sigma * root + self.c * sigma / root   # Alg. 1: mu_n + c * beta_n
        return top, float(scores[top]), tau

    def check(self, scores, ids):
        top, s_max, tau = self.threshold(scores)
        return {ids[top]} if s_max > tau else set()


class SessionDefense:
    """Condition C (ours): session-level repetition defense.

    Remembers the top-1 document of the last `window` queries in this session. Once one
    document has been the top match in at least `k` of them, it is hidden for the rest of
    the session. One sharp question is normal; the same document coming up again and again
    is what a membership-inference attacker produces.
    """
    name = "session"

    def __init__(self, window=C.SESSION_WINDOW, k=C.SESSION_K):
        self.k = k
        self.recent = deque(maxlen=window)    # top-1 doc id of each recent query
        self.hidden = set()                   # documents hidden for the rest of the session

    def check(self, scores, ids):
        top = ids[int(np.argmax(scores))]     # raw top-1, even if it is already hidden
        self.recent.append(top)
        for doc, count in Counter(self.recent).items():
            if count >= self.k:
                self.hidden.add(doc)
        return set(self.hidden)


DEFENSES = {d.name: d for d in (NoDefense, Mirabel, SessionDefense)}


def make_defense(name, **kwargs):
    return DEFENSES[name](**kwargs)