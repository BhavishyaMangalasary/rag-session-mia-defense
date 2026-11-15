"""Defenses that sit inside the RAG server, between retrieval and generation.

Every defense has the same interface:
    check(scores, ids) -> set of doc ids to hide from the LLM for this query
where `scores` are the query's cosine similarities to every document in the store and
`ids` are the matching document ids. One defense object is created per session, so a
defense may remember earlier queries (the session-level defense will; Mirabel does not).
"""
import math

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


DEFENSES = {d.name: d for d in (NoDefense, Mirabel)}


def make_defense(name):
    return DEFENSES[name]()