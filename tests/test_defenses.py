"""Unit tests for the defenses. No models or data needed, so they run anywhere (and in CI)."""
import numpy as np

from ragmia.defenses import Mirabel, NoDefense, SessionDefense, make_defense

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