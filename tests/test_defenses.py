"""Unit tests for the defenses. No models or data needed, so they run anywhere (and in CI)."""
import numpy as np

from ragmia.defenses import Mirabel, MirabelSession, NoDefense, SessionDefense, make_defense

IDS = [f"doc{i}" for i in range(10)]


def aimed_at(i):
    """Similarity scores for a query whose best match is document i."""
    scores = np.full(len(IDS), 0.1)
    scores[i] = 0.9
    return scores


def run(defense, targets):
    """Send one query per target index; return what was hidden at each step."""
    return [defense.check(aimed_at(i), IDS) for i in targets]


def test_no_defense_never_hides():
    assert run(NoDefense(), [0, 0, 0, 0, 0]) == [set()] * 5


def test_session_hides_from_the_kth_repeat():
    hidden = run(SessionDefense(window=5, k=3), [0, 0, 0, 0, 0])
    assert hidden == [set(), set(), {"doc0"}, {"doc0"}, {"doc0"}]


def test_session_ignores_normal_browsing():
    assert run(SessionDefense(window=5, k=3), [0, 1, 2, 3, 4, 5, 6]) == [set()] * 7


def test_session_catches_interleaved_repeats():
    hidden = run(SessionDefense(window=5, k=3), [0, 1, 0, 2, 0])
    assert hidden[-1] == {"doc0"} and hidden[3] == set()


def test_session_hiding_lasts_for_the_whole_session():
    hidden = run(SessionDefense(window=5, k=3), [0, 0, 0, 1, 2, 3, 4, 5])
    assert all(h == {"doc0"} for h in hidden[2:])


def test_session_window_forgets_old_queries():
    # doc0 twice, then five other documents, then doc0 again: only 1 of the last 5
    hidden = run(SessionDefense(window=5, k=3), [0, 0, 1, 2, 3, 4, 5, 0])
    assert hidden[-1] == set()


def test_session_parameters_are_configurable():
    hidden = run(make_defense("session", window=3, k=2), [0, 0])
    assert hidden == [set(), {"doc0"}]


def test_mirabel_flags_a_similarity_spike_only():
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(3000)]
    flat = rng.normal(0.2, 0.05, len(ids))
    spike = flat.copy()
    spike[7] = 0.95
    assert Mirabel().check(flat, ids) == set()
    assert Mirabel().check(spike, ids) == {"d7"}



def spike_at(i, n=3000):
    """Similarity scores with an extreme outlier at document i (what Mirabel flags)."""
    scores = np.random.default_rng(i).normal(0.2, 0.05, n)
    scores[i] = 0.95
    return scores


def flat(seed, n=3000):
    """Similarity scores with no outlier; best match is whichever document happens to be highest."""
    return np.random.default_rng(100 + seed).normal(0.2, 0.05, n)


def test_mirabel_session_keeps_mirabel_flags_for_the_whole_session():
    ids = [f"d{i}" for i in range(3000)]
    d = MirabelSession()
    first = d.check(spike_at(7), ids)                 # Mirabel flags d7
    later = [d.check(flat(s), ids) for s in range(3)]  # unflagged queries
    assert first == {"d7"} and all("d7" in h for h in later)
    assert Mirabel().check(flat(0), ids) == set()     # plain Mirabel would have forgotten it


def test_mirabel_session_also_applies_the_repetition_rule():
    ids = [f"d{i}" for i in range(3000)]
    queries = []
    for s in range(3):                                 # d5 is the top match each time, but no spike
        scores = flat(s)
        scores[5] = scores.max() + 0.001
        queries.append(scores)
    assert all(Mirabel().check(q, ids) == set() for q in queries)
    d = MirabelSession(window=5, k=3)
    hidden = [d.check(q, ids) for q in queries]
    assert hidden == [set(), set(), {"d5"}]